#!/usr/bin/env python3
"""
OntoVis - AI-powered 3D volume analysis and visualization

Main entry point for the OntoVis application.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ontovis.run_manager import RunManager, RunRegistry
from ontovis import __version__


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OntoVis - AI-powered 3D volume analysis and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start interactive CLI (default)
  %(prog)s --list-runs              # List all runs
  %(prog)s --latest                 # Show latest run info
  %(prog)s --run-name my_analysis   # Start with custom run name
  %(prog)s --cleanup 10             # Keep only 10 most recent runs

For more information, see docs/ directory.
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'OntoVis {__version__}'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        help='Custom name for this run (default: timestamp)'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='runs',
        help='Base directory for runs (default: runs)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    # Run management commands
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List all runs and exit'
    )

    parser.add_argument(
        '--latest',
        action='store_true',
        help='Show latest run info and exit'
    )

    parser.add_argument(
        '--cleanup',
        type=int,
        metavar='N',
        help='Remove old runs, keeping only N most recent'
    )

    args = parser.parse_args()

    # Handle run management commands
    if args.list_runs or args.latest or args.cleanup:
        registry = RunRegistry(base_dir=args.base_dir)

        if args.list_runs:
            registry.print_summary()
            return 0

        if args.latest:
            latest = registry.get_latest_run()
            if latest:
                print(f"\nLatest run: {latest['run_id']}")
                print(f"Status: {latest.get('status', 'unknown')}")
                print(f"Start: {latest['start_time']}")
                if 'end_time' in latest:
                    print(f"End: {latest['end_time']}")
                    print(f"Duration: {latest['duration_seconds']:.2f}s")
                print(f"Directory: {latest['run_dir']}")
                print()
            else:
                print("No runs found.")
            return 0

        if args.cleanup:
            registry.cleanup_old_runs(keep_n=args.cleanup)
            return 0

    # Start interactive CLI
    print("="*70)
    print(f"OntoVis v{__version__}")
    print("AI-powered 3D volume analysis and visualization")
    print("="*70)
    print()

    # Import here to avoid slow startup for --help, etc.
    from ontovis.interactive_cli import run_interactive_cli

    try:
        run_interactive_cli(
            run_name=args.run_name,
            base_dir=args.base_dir,
            log_level=args.log_level
        )
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        return 130

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
