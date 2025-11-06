#!/usr/bin/env python3
"""
Render environment variable templates in .yaml/.yml files within a folder (non-recursive).

Replaces occurrences of {{ENV_VAR}} and ${{ENV_VAR}} with the value of the
environment variable ENV_VAR, if defined. Only simple single-word names are
considered valid (letters, digits, underscore; must start with a letter or underscore).
Only files with extension .yaml or .yml are processed; other files are ignored.

Usage:
  python quickstarts/resolve_env_templates.py /path/to/folder

The script creates a temporary directory, writes processed files to it using the
same base filenames, and prints the temporary directory path to stdout.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable


# Matches either {{ENV_NAME}} or ${{ENV_NAME}}, where ENV_NAME is a valid env var name
_TEMPLATE_PATTERN = re.compile(r"(\$)?\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _replace_env_templates(text: str, get_env: Callable[[str], str | None]) -> str:
    """Replace template tokens in the given text using provided env lookup.

    - Templates supported: {{ENV}} and ${{ENV}}
    - Only replaces when ENV exists in env map
    - Leaves unknown templates unchanged
    """

    def _sub_fn(match: re.Match[str]) -> str:
        # match groups: (optional '$', ENV_NAME)
        env_name = match.group(2)
        env_value = get_env(env_name)
        if env_value is None:
            return match.group(0)
        return env_value

    return _TEMPLATE_PATTERN.sub(_sub_fn, text)


def render_folder_with_env_templates(source_dir: str | Path) -> str:
    """Process files in source_dir (non-recursive), writing results to a temp dir.

    Returns the path to the temporary directory.
    """
    src_path = Path(source_dir).resolve()
    if not src_path.exists() or not src_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {src_path}")

    temp_dir = tempfile.mkdtemp(prefix="rendered_components_")
    dst_path = Path(temp_dir)

    # iterate non-recursively
    for entry in sorted(src_path.iterdir()):
        if entry.is_dir():
            # Skip subdirectories (non-recursive as requested)
            continue

        # Only process YAML files
        if entry.suffix.lower() not in {".yaml", ".yml"}:
            continue

        dest_file = dst_path / entry.name

        # Read as UTF-8 text (YAML is text). If reading fails, copy as-is.
        try:
            text = entry.read_text(encoding="utf-8")
        except Exception:
            shutil.copy2(entry, dest_file)
            continue

        replaced = _replace_env_templates(text, os.getenv)

        if replaced == text:
            # No changes - copy file verbatim preserving metadata
            shutil.copy2(entry, dest_file)
        else:
            # Write processed content
            dest_file.write_text(replaced, encoding="utf-8")
            # Best-effort preserve mode/timestamps
            try:
                shutil.copystat(entry, dest_file)
            except Exception:
                pass

    return str(dst_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render env templates in a folder.")
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing files to process (non-recursive)",
    )

    args = parser.parse_args(argv)

    try:
        output_dir = render_folder_with_env_templates(args.folder)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
