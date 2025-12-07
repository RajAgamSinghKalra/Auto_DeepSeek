# AutoDeepSeek

AutoDeepSeek is an autonomous assistant powered by the DeepSeek-R1 7B model. The agent can create and run code, manage files and browse the web to complete tasks you describe. It now includes a cross-platform terminal UI that detects your OS/hardware, guides you through setup, and lets you pick which LLM to install and run. It works on Linux, Windows, and macOS with GPU acceleration (CUDA, ROCm, or Apple Silicon) when available, and will fall back to CPU if needed.

## Features

- Breaks down complex tasks into simple steps
- Writes and executes Python or shell scripts in a workspace
- Searches the web and opens pages with Firefox
- Reads and modifies files you create
- Optional full system access for file and command operations
- Runs safe system commands
- Keeps logs of every session

## Quick Start

1. Clone this repository and enter the folder:
   ```bash
   git clone https://github.com/yourusername/autodeepseek.git
   cd autodeepseek
   ```
2. Launch the new TUI (this is now the first thing to run):
   ```bash
   ./run_autodeepseek.sh
   ```
   The TUI will:
   - Detect your OS and hardware (Linux/Windows/macOS, CPU/GPU type).
   - Offer setup commands tailored to your platform (e.g., apt, Homebrew, or Chocolatey). Nothing installs unless you choose it.
   - Show an indicator for missing dependencies.
   - Let you pick an LLM from a curated list or enter a custom ID, then optionally download it on demand.
3. From the TUI, choose **Launch AutoDeepSeek agent** after your environment is ready and a model is selected.

## Using the Agent

When the agent starts you will see a prompt. Type a task in plain language, for example:

```
Create a Python script that prints numbers from 1 to 10.
```

AutoDeepSeek will plan the steps, write files and run code until the task is done. By default all files are saved in the `autodeepseek_workspace` folder. Use `exit` to stop the program.

> **Note**
> File operations can be restricted to the workspace (default) or unlocked for full system access.
> Enable unrestricted mode with the environment variable `FULL_ACCESS=true` or pass `--full-access` when launching.
> When unrestricted and no path is provided, files are saved to your `~/Desktop`.

### Optional GUI

A simple graphical interface is available:

```bash
python3 gui.py
```

Type a task in the box and press **Run**. Press **Abort** to stop the current task.

### Testing

To make sure everything works you can run:

```bash
python3 test_autodeepseek.py
```

This launches a few basic checks of the main features.

### Troubleshooting

- **NumPy compatibility errors**: If you see a warning like
  `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`,
  downgrade NumPy with `pip install 'numpy<2.0'`.
- **Browser initialization failure**: The agent uses Firefox via Selenium.
  Make sure Firefox is installed and working if you want browsing features.
