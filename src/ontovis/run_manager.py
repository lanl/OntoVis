"""Run management system for OntoVis.

Manages:
- Unique run IDs
- Run directories for artifacts
- Structured logging
- Run metadata
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import sys


class RunManager:
    """Manages a single run with logging and artifact storage."""

    def __init__(
        self,
        run_name: Optional[str] = None,
        base_dir: str = "runs",
        log_level: int = logging.INFO
    ):
        """Initialize a new run.

        Args:
            run_name: Optional custom run name. If None, auto-generates timestamp-based name
            base_dir: Base directory for all runs (default: "runs")
            log_level: Logging level (default: INFO)
        """
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            self.run_id = f"{timestamp}_{run_name}"
        else:
            self.run_id = timestamp

        # Setup directories
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)

        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        self.renders_dir = self.run_dir / "renders"
        self.renders_dir.mkdir(exist_ok=True)

        self.histograms_dir = self.run_dir / "histograms"
        self.histograms_dir.mkdir(exist_ok=True)

        # Setup logging
        self.log_file = self.logs_dir / "run.log"
        self._setup_logging(log_level)

        # Metadata
        self.metadata = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "base_dir": str(self.base_dir.absolute()),
            "run_dir": str(self.run_dir.absolute()),
            "status": "running"
        }

        self._save_metadata()

        self.logger.info("="*70)
        self.logger.info(f"Run initialized: {self.run_id}")
        self.logger.info(f"Run directory: {self.run_dir.absolute()}")
        self.logger.info("="*70)

    def _setup_logging(self, log_level: int):
        """Setup logging to file and console."""
        # Create logger
        self.logger = logging.getLogger(f"ontovis.{self.run_id}")
        self.logger.setLevel(log_level)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler (detailed)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler (less verbose)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def _save_metadata(self):
        """Save run metadata to file."""
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def update_metadata(self, key: str, value: Any):
        """Update metadata and save.

        Args:
            key: Metadata key
            value: Value to store
        """
        self.metadata[key] = value
        self._save_metadata()

    def get_artifact_path(self, filename: str, subdir: Optional[str] = None) -> Path:
        """Get path for storing an artifact.

        Args:
            filename: Name of the artifact file
            subdir: Optional subdirectory within artifacts (e.g., "renders", "histograms")

        Returns:
            Path: Full path for the artifact
        """
        if subdir:
            artifact_dir = self.artifacts_dir / subdir
            artifact_dir.mkdir(exist_ok=True)
            return artifact_dir / filename
        return self.artifacts_dir / filename

    def get_render_path(self, filename: str) -> Path:
        """Get path for storing a render.

        Args:
            filename: Name of the render file

        Returns:
            Path: Full path for the render
        """
        return self.renders_dir / filename

    def get_histogram_path(self, filename: str) -> Path:
        """Get path for storing a histogram.

        Args:
            filename: Name of the histogram file

        Returns:
            Path: Full path for the histogram
        """
        return self.histograms_dir / filename

    def log_operation(
        self,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
        level: str = "info"
    ):
        """Log an operation with structured details.

        Args:
            operation: Name of the operation
            details: Optional dictionary of operation details
            level: Log level ("debug", "info", "warning", "error")
        """
        log_method = getattr(self.logger, level.lower())

        log_method(f"Operation: {operation}")
        if details:
            for key, value in details.items():
                log_method(f"  {key}: {value}")

        # Also save to operations log
        operations_file = self.logs_dir / "operations.jsonl"
        operation_record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details or {},
            "level": level
        }

        with open(operations_file, 'a') as f:
            f.write(json.dumps(operation_record) + '\n')

    def finalize(self, status: str = "completed", summary: Optional[Dict[str, Any]] = None):
        """Finalize the run.

        Args:
            status: Final status ("completed", "failed", "interrupted")
            summary: Optional summary dictionary
        """
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["status"] = status

        if summary:
            self.metadata["summary"] = summary

        # Calculate duration
        start = datetime.fromisoformat(self.metadata["start_time"])
        end = datetime.fromisoformat(self.metadata["end_time"])
        duration = (end - start).total_seconds()
        self.metadata["duration_seconds"] = duration

        self._save_metadata()

        self.logger.info("="*70)
        self.logger.info(f"Run finalized: {status}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        if summary:
            self.logger.info("Summary:")
            for key, value in summary.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("="*70)

        # Close handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def create_summary_report(self) -> str:
        """Create a human-readable summary report.

        Returns:
            str: Formatted summary report
        """
        report_lines = [
            "="*70,
            f"Run Summary: {self.run_id}",
            "="*70,
            "",
            f"Status: {self.metadata['status']}",
            f"Start: {self.metadata['start_time']}",
        ]

        if "end_time" in self.metadata:
            report_lines.append(f"End: {self.metadata['end_time']}")
            report_lines.append(f"Duration: {self.metadata['duration_seconds']:.2f}s")

        report_lines.append("")
        report_lines.append("Directories:")
        report_lines.append(f"  Logs: {self.logs_dir.absolute()}")
        report_lines.append(f"  Artifacts: {self.artifacts_dir.absolute()}")
        report_lines.append(f"  Renders: {self.renders_dir.absolute()}")
        report_lines.append(f"  Histograms: {self.histograms_dir.absolute()}")

        if "summary" in self.metadata:
            report_lines.append("")
            report_lines.append("Summary:")
            for key, value in self.metadata["summary"].items():
                report_lines.append(f"  {key}: {value}")

        report_lines.append("")
        report_lines.append("="*70)

        report = "\n".join(report_lines)

        # Save to file
        summary_file = self.run_dir / "SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write(report)

        return report


class RunRegistry:
    """Registry of all runs for querying and management."""

    def __init__(self, base_dir: str = "runs"):
        """Initialize run registry.

        Args:
            base_dir: Base directory containing all runs
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def list_runs(self, status: Optional[str] = None) -> list:
        """List all runs, optionally filtered by status.

        Args:
            status: Optional status filter ("running", "completed", "failed")

        Returns:
            List of run metadata dictionaries
        """
        runs = []

        for run_dir in sorted(self.base_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            metadata_file = run_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if status is None or metadata.get("status") == status:
                runs.append(metadata)

        return runs

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific run.

        Args:
            run_id: Run ID to retrieve

        Returns:
            Run metadata dictionary or None if not found
        """
        run_dir = self.base_dir / run_id
        metadata_file = run_dir / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r') as f:
            return json.load(f)

    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        """Get the most recent run.

        Returns:
            Run metadata dictionary or None if no runs exist
        """
        runs = self.list_runs()
        if not runs:
            return None

        # Sort by start_time
        runs.sort(key=lambda x: x["start_time"], reverse=True)
        return runs[0]

    def cleanup_old_runs(self, keep_n: int = 10):
        """Remove old runs, keeping only the N most recent.

        Args:
            keep_n: Number of runs to keep
        """
        runs = self.list_runs()
        runs.sort(key=lambda x: x["start_time"], reverse=True)

        for run in runs[keep_n:]:
            run_dir = Path(run["run_dir"])
            if run_dir.exists():
                import shutil
                shutil.rmtree(run_dir)
                print(f"Removed old run: {run['run_id']}")

    def print_summary(self):
        """Print summary of all runs."""
        runs = self.list_runs()

        print("\n" + "="*70)
        print("Run Registry Summary")
        print("="*70)
        print(f"\nTotal runs: {len(runs)}")

        # Group by status
        status_counts = {}
        for run in runs:
            status = run.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        print("\nBy status:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")

        print("\nRecent runs:")
        runs.sort(key=lambda x: x["start_time"], reverse=True)
        for run in runs[:5]:
            print(f"  {run['run_id']}: {run.get('status', 'unknown')}")

        print("\n" + "="*70 + "\n")
