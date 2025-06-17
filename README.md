# AutoDeepSeek

AutoDeepSeek is an autonomous assistant powered by the DeepSeek-R1 7B model. The agent can create and run code, manage files and browse the web to complete tasks you describe. It works best on Linux with an AMD GPU (ROCm), but it can fall back to CPU if needed.

## Features

- Breaks down complex tasks into simple steps
- Writes and executes Python or shell scripts in a workspace
- Searches the web and opens pages with Firefox
- Reads and modifies files you create
- Runs safe system commands
- Keeps logs of every session

## Quick Start

1. Install Linux (Ubuntu 20.04 or later is recommended). Having an AMD GPU speeds up the model but is optional.
2. Clone this repository and enter the folder:
   ```bash
   git clone https://github.com/yourusername/autodeepseek.git
   cd autodeepseek
   ```
3. Run the setup script:
   ```bash
   python3 setup.py
   ```
   The script creates a virtual environment, installs dependencies, prepares Firefox and ROCm, and writes a `.env` configuration file. During this step you may be asked for your password to install system packages.
4. Reboot your computer if the script tells you to.
5. Start the agent:
   ```bash
   ./run_autodeepseek.sh
   ```

## Using the Agent

When the launcher starts you will see a prompt. Type a task in plain language, for example:

```
Create a Python script that prints numbers from 1 to 10.
```

AutoDeepSeek will plan the steps, write files and run code until the task is done. All files are saved in the `autodeepseek_workspace` folder. Use `exit` to stop the program.

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
