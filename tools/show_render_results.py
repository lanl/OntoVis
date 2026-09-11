#!/usr/bin/env python3
"""
Display recent render results in an easy-to-see format.

Usage:
    python tools/show_render_results.py
    python tools/show_render_results.py --run-dir runs/20260910_164112
    python tools/show_render_results.py --count 20
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontovis.run_manager import RunManager


def show_renders(run_dir: str = None, count: int = 10):
    """Show recent renders from a run directory.

    Args:
        run_dir: Specific run directory, or None for current
        count: Number of renders to show
    """
    if run_dir:
        run_path = Path(run_dir)
        if not run_path.exists():
            print(f"❌ Run directory not found: {run_dir}")
            sys.exit(1)
        renders_dir = run_path / "renders"
    else:
        run_mgr = RunManager()
        renders_dir = run_mgr.renders_dir

    if not renders_dir.exists():
        print(f"❌ Renders directory not found: {renders_dir}")
        sys.exit(1)

    # Find all PNG files (excluding iterations subdirectory)
    png_files = []
    for png_file in renders_dir.glob("*.png"):
        if png_file.is_file():
            png_files.append(png_file)

    if not png_files:
        print(f"No renders found in {renders_dir}")
        print(f"\nChecking iteration folders...")

        # Check for iteration folders
        iterations_dir = renders_dir / "iterations"
        if iterations_dir.exists():
            iteration_folders = sorted(iterations_dir.iterdir())
            print(f"Found {len(iteration_folders)} iteration folders:\n")

            for folder in iteration_folders[-5:]:  # Show last 5
                if folder.is_dir():
                    pngs = list(folder.glob("*.png"))
                    timestamp = folder.name.split('_')[-2] + '_' + folder.name.split('_')[-1]
                    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    print(f"📁 {folder.name}")
                    print(f"   Time: {dt.strftime('%H:%M:%S')}")
                    print(f"   Images: {len(pngs)}")
                    print(f"   Path: {folder}")
                    print()
        return

    # Sort by modification time (newest first)
    png_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # Take the N most recent
    recent = png_files[:count]

    print("="*80)
    print(f"RECENT RENDERS ({len(recent)} of {len(png_files)} total)")
    print("="*80)
    print(f"Location: {renders_dir}\n")

    for i, render_path in enumerate(recent, 1):
        stat = render_path.stat()
        size_kb = stat.st_size / 1024
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        print(f"{i}. {render_path.name}")
        print(f"   📅 Created: {mod_time}")
        print(f"   📦 Size: {size_kb:.1f} KB")
        print(f"   📂 Full path: {render_path}")
        print()

    # Check for iteration folders
    iterations_dir = renders_dir / "iterations"
    if iterations_dir.exists():
        iteration_folders = list(iterations_dir.iterdir())
        if iteration_folders:
            print("="*80)
            print(f"ITERATION FOLDERS ({len(iteration_folders)} total)")
            print("="*80)
            print()

            # Show most recent 3
            for folder in sorted(iteration_folders)[-3:]:
                if folder.is_dir():
                    pngs = list(folder.glob("*.png"))
                    readme = folder / "README.md"
                    summary = folder / "SUMMARY.json"

                    timestamp = folder.name.split('_')[-2] + '_' + folder.name.split('_')[-1]
                    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

                    print(f"📁 {folder.name}")
                    print(f"   Time: {dt.strftime('%H:%M:%S')}")
                    print(f"   Images: {len(pngs)}")
                    if readme.exists():
                        print(f"   ✓ README.md")
                    if summary.exists():
                        print(f"   ✓ SUMMARY.json")
                    print(f"   Path: {folder}")
                    print()

    print("="*80)
    print("\nTip: To analyze a render, use:")
    print(f"  analyze_image(image_path=\"{recent[0]}\")")


def main():
    parser = argparse.ArgumentParser(
        description="Display recent render results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                      # Show current run
  %(prog)s --run-dir runs/20260910_164112       # Specific run
  %(prog)s --count 20                           # Show more renders
        """
    )

    parser.add_argument(
        '--run-dir',
        help='Specific run directory to inspect'
    )

    parser.add_argument(
        '--count',
        '-n',
        type=int,
        default=10,
        help='Number of renders to show (default: 10)'
    )

    args = parser.parse_args()

    show_renders(run_dir=args.run_dir, count=args.count)


if __name__ == "__main__":
    main()
