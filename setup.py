#!/usr/bin/env python3
"""
Setup entrypoint for AutoDeepSeek.

The project now uses the interactive TUI as the primary way to detect
your OS/hardware, run optional setup steps, and select models. Running
this script simply forwards to the TUI so existing workflows continue to
function.
"""
from __future__ import annotations

import sys

import tui_launcher


def main() -> int:
    print("AutoDeepSeek setup is now interactive.")
    print("Launching the TUI to guide you through environment preparation...")
    return tui_launcher.main()


if __name__ == "__main__":
    sys.exit(main())
