#!/bin/bash
# AutoDeepSeek unified launcher for the interactive TUI

set -e

if [ ! -f "tui_launcher.py" ]; then
    echo "tui_launcher.py not found. Run this script from the project root." >&2
    exit 1
fi

if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    if [ "$(uname -s)" = "MINGW" ] || [[ "$OS" == "Windows_NT" ]]; then
        # shellcheck disable=SC1091
        source venv/Scripts/activate
    else
        # shellcheck disable=SC1091
        source venv/bin/activate
    fi
else
    echo "No virtual environment detected. The TUI can help you create one."
fi

python3 tui_launcher.py "$@"
