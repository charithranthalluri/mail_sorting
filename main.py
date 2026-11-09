"""Mail Sorter — Main entry point and orchestration loop.

This module wires all components together and runs the continuous
email monitoring pipeline. It coordinates:
  Email Watcher → Reader → Parser → Cleaner → Classifier → Validator →
  Decision Engine → Action Executor → Database

The main loop remains thin — all business logic lives in the appropriate modules.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import AppConfig, load_config
from utils.logger import setup_logging, get_logger, set_trace_id, clear_trace_id

logger = get_logger("main")


class GracefulShutdown:
    """Handle SIGINT/SIGTERM for graceful shutdown of the polling loop."""

    def __init__(self) -> None:
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: object) -> None:
        logger.info("Shutdown signal received (signal=%d). Finishing current cycle...", signum)
        self._shutdown_requested = True

    @property
    def should_stop(self) -> bool:
        return self._shutdown_requested


class MailSorterApp:
    """Main application orchestrator.

    Wires together all components and runs the processing pipeline.
    Each email is processed independently — one bad email cannot crash
    the entire system.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._shutdown = GracefulShutdown()

        # Lazy-initialized components
        self._gmail_service = None
        self._reader = None
        self._parser = None
        self._cleaner = None
        self._classifier = None
        self._decision_engine = None
        self._action_executor = None
        self._database = None
        self._label_id_map: dict[str, str] = {}
        self._start_time: Optional[datetime] = None

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        from gmail.auth import get_gmail_service
        from gmail.reader import GmailReader
        from gmail.actions import GmailActionExecutor
        from processing.parser import EmailParser
        from processing.cleaner import EmailCleaner
        from ai.classifier import get_classifier
        from engine.decision_engine import DecisionEngine
        from storage.database import Database

        logger.info("Initializing components...")

        # Gmail
        self._gmail_service = get_gmail_service(self.config.gmail)
        self._reader = GmailReader(self._gmail_service, self.config.gmail)
        self._action_executor = GmailActionExecutor(
            self._gmail_service, self.config.gmail, self.config.labels,
            dry_run=self.config.dry_run,
        )

        # Processing
        self._parser = EmailParser()
        self._cleaner = EmailCleaner()

        # AI
        self._classifier = get_classifier(self.config.llm)

        # Engine
        self._decision_engine = DecisionEngine(self.config.labels)

        # Storage
        self._database = Database(self.config.database.path)

        # Record start time for first-run lookback control
        self._start_time = datetime.now(timezone.utc)

        # Ensure all required Gmail labels exist
        all_labels = self.config.labels.all_labels()
        self._label_id_map = self._action_executor.ensure_labels_exist(all_labels)
        logger.info(
            "Initialized %d labels, dry_run=%s", len(self._label_id_map), self.config.dry_run
        )

        mode = "DRY-RUN" if self.config.dry_run else "LIVE"
        logger.info("Mail Sorter initialized in %s mode", mode)

    def _get_after_timestamp(self) -> Optional[datetime]:
        """Determine the 'after' timestamp for email queries.

        On first run, respects INITIAL_LOOKBACK_HOURS.
        On subsequent runs, uses the start time to avoid reprocessing.
        """
        if self.config.initial_lookback_hours > 0 and self._start_time:
            return self._start_time - timedelta(hours=self.config.initial_lookback_hours)
        return self._start_time

    def _process_single_email(self, message_id: str) -> None:
        """Process a single email through the full pipeline.

        Each email gets its own trace ID for log correlation.
        Failures are caught and recorded — they never crash the loop.
        """
        trace_id = set_trace_id()
        logger.info("Processing email message_id=%s", message_id)

        try:
            # Skip already-processed emails
            if self._database.is_processed(message_id):
                logger.debug("Skipping already-processed email message_id=%s", message_id)
                return

            # Record discovery if not already tracked
            if not self._database.has_record(message_id):
                from storage.models import ProcessingRecord
                record = ProcessingRecord(message_id=message_id)
                self._database.insert_record(record)

            # --- PHASE 1: Retrieve ---
            raw_message = self._reader.get_message(message_id)
            if not raw_message:
                self._database.mark_failed(message_id, "Failed to retrieve message from Gmail")
                return

            # --- PHASE 2: Parse ---
            email_msg = self._parser.parse(raw_message)
            self._database.update_status(
                message_id, 
                __import__("storage.models", fromlist=["ProcessingStatus"]).ProcessingStatus.PARSED
            )
            logger.info(
                "Parsed email: subject=%s, sender=%s",
                email_msg.subject[:60] if email_msg.subject else "(no subject)",
                email_msg.sender[:40] if email_msg.sender else "(unknown)",
            )

            # --- PHASE 3: Clean ---
            cleaned_body = self._cleaner.clean(
                email_msg.raw_body or email_msg.clean_body,
                is_html=email_msg.has_html and not email_msg.raw_body,
            )
            email_msg = email_msg.model_copy(update={"clean_body": cleaned_body})

            # Update thread_id in database
            if email_msg.thread_id:
                # Update thread_id via a direct DB call if needed
                pass

            # --- PHASE 4: Classify ---
            logger.info("Classifying email with %s...", self._classifier.provider_name)
            classification = self._classifier.classify(email_msg)
            from storage.models import ProcessingStatus
            self._database.update_status(message_id, ProcessingStatus.CLASSIFIED)
            self._database.update_classification(
                message_id=message_id,
                category=classification.category.value,
                priority=classification.priority.value,
                urgency=classification.urgency.value,
                action_required=classification.action_required,
                deadline=classification.deadline,
                summary=classification.summary,
                confidence=classification.confidence,
            )
            logger.info(
                "Classified: category=%s, priority=%s, urgency=%s, confidence=%.2f",
                classification.category.value,
                classification.priority.value,
                classification.urgency.value,
                classification.confidence,
            )

            # --- PHASE 5: Validate (already done by Pydantic in classifier) ---
            self._database.update_status(message_id, ProcessingStatus.VALIDATED)

            # --- PHASE 6: Decide ---
            actions = self._decision_engine.decide(classification)
            logger.info("Decision engine produced %d actions", len(actions))

            # --- PHASE 7: Execute or Dry-Run ---
            if self.config.dry_run:
                report = self._decision_engine.format_dry_run_report(
                    email_msg.subject, classification, actions
                )
                logger.info("DRY-RUN:\n%s", report)
            else:
                self._execute_actions(message_id, actions)

            # Record actions taken
            action_descriptions = [
                {"type": a.action_type, "label": a.label_name, "reason": a.reason}
                for a in actions
            ]
            self._database.update_actions(message_id, json.dumps(action_descriptions))
            self._database.update_status(message_id, ProcessingStatus.ACTIONED)

            # --- PHASE 8: Complete ---
            self._database.mark_complete(message_id)
            logger.info("Completed processing message_id=%s", message_id)

        except Exception as e:
            logger.error(
                "Failed to process message_id=%s: %s: %s",
                message_id, type(e).__name__, e,
                exc_info=True,
            )
            try:
                self._database.mark_failed(message_id, f"{type(e).__name__}: {e}")
            except Exception as db_err:
                logger.error("Failed to record failure in database: %s", db_err)
        finally:
            clear_trace_id()

    def _execute_actions(self, message_id: str, actions: list) -> None:
        """Execute Gmail actions for a processed email.

        Each action is attempted independently — one failing action
        does not prevent others from executing.
        """
        for action in actions:
            try:
                success = False
                if action.action_type == "add_label" and action.label_name:
                    label_id = self._label_id_map.get(action.label_name)
                    if label_id:
                        success = self._action_executor.add_label(message_id, label_id)
                    else:
                        logger.warning(
                            "Label ID not found for '%s'", action.label_name
                        )
                elif action.action_type == "star":
                    success = self._action_executor.star_message(message_id)
                elif action.action_type == "unstar":
                    success = self._action_executor.unstar_message(message_id)
                elif action.action_type == "archive":
                    success = self._action_executor.archive_message(message_id)
                elif action.action_type == "mark_read":
                    success = self._action_executor.mark_read(message_id)
                elif action.action_type == "mark_unread":
                    success = self._action_executor.mark_unread(message_id)
                elif action.action_type == "remove_label" and action.label_name:
                    label_id = self._label_id_map.get(action.label_name)
                    if label_id:
                        success = self._action_executor.remove_label(message_id, label_id)
                else:
                    logger.warning("Unknown action type: %s", action.action_type)

                if success:
                    logger.info(
                        "Action succeeded: %s (label=%s) for message_id=%s",
                        action.action_type, action.label_name, message_id,
                    )
                else:
                    logger.warning(
                        "Action failed: %s (label=%s) for message_id=%s",
                        action.action_type, action.label_name, message_id,
                    )

            except Exception as e:
                logger.error(
                    "Error executing action %s for message_id=%s: %s",
                    action.action_type, message_id, e,
                )

    def run_once(self) -> int:
        """Run a single polling cycle.

        Returns:
            Number of emails processed in this cycle.
        """
        after_ts = self._get_after_timestamp()
        logger.debug("Fetching new emails after %s", after_ts)

        try:
            message_ids = self._reader.fetch_new_message_ids(
                query="is:unread",
                after_timestamp=after_ts,
            )
        except Exception as e:
            logger.error("Failed to fetch message IDs: %s", e)
            return 0

        if not message_ids:
            logger.debug("No new emails found")
            return 0

        logger.info("Found %d candidate emails", len(message_ids))
        processed = 0

        for msg_id in message_ids:
            if self._shutdown.should_stop:
                logger.info("Shutdown requested, stopping mid-cycle")
                break
            self._process_single_email(msg_id)
            processed += 1

        return processed

    def run(self) -> None:
        """Run the continuous polling loop.

        Polls for new emails at the configured interval.
        Handles graceful shutdown on SIGINT/SIGTERM.
        """
        self._init_components()

        logger.info(
            "Starting mail sorter loop (poll_interval=%ds, dry_run=%s)",
            self.config.poll_interval_seconds,
            self.config.dry_run,
        )

        cycle = 0
        while not self._shutdown.should_stop:
            cycle += 1
            logger.info("=== Polling cycle %d ===", cycle)

            try:
                start = time.monotonic()
                count = self.run_once()
                elapsed = time.monotonic() - start
                logger.info(
                    "Cycle %d complete: processed %d emails in %.1fs",
                    cycle, count, elapsed,
                )

                # Log stats periodically
                if cycle % 10 == 0:
                    stats = self._database.get_stats()
                    logger.info("Database stats: %s", stats)

            except Exception as e:
                logger.error("Error in polling cycle %d: %s", cycle, e, exc_info=True)

            # Sleep until next cycle, checking for shutdown every second
            if not self._shutdown.should_stop:
                logger.debug(
                    "Sleeping %d seconds until next cycle...",
                    self.config.poll_interval_seconds,
                )
                for _ in range(self.config.poll_interval_seconds):
                    if self._shutdown.should_stop:
                        break
                    time.sleep(1)

        logger.info("Mail sorter shutdown complete")


def main() -> None:
    """Application entry point."""
    config = load_config()

    # Setup logging
    setup_logging(level=config.log_level, log_file=config.log_file)

    logger.info("=" * 60)
    logger.info("Mail Sorter starting")
    logger.info("  Dry-run: %s", config.dry_run)
    logger.info("  Poll interval: %ds", config.poll_interval_seconds)
    logger.info("  Lookback hours: %d", config.initial_lookback_hours)
    logger.info("  LLM model: %s", config.llm.model)
    logger.info("  Database: %s", config.database.path)
    logger.info("=" * 60)

    app = MailSorterApp(config)
    app.run()


if __name__ == "__main__":
    main()
