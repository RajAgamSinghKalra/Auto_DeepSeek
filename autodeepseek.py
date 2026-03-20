#!/usr/bin/env python3
"""
AutoDeepSeek - Autonomous AI Agent
Main agent orchestrator with full system access capabilities
"""

import os
import re
import sys
import json
import time
import shlex
import shutil
import subprocess
import tempfile
import threading
import platform
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

# Third-party imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import requests
    import psutil
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# Optional: DirectML for AMD/Intel GPUs on Windows
try:
    import torch_directml
    _DIRECTML_AVAILABLE = True
except ImportError:
    _DIRECTML_AVAILABLE = False

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


SYSTEM_PROMPT = """\
You are AutoDeepSeek, an autonomous AI agent with full system access. You can:
1. Write, read, and manage files and directories
2. Execute code in Python, Bash, or JavaScript
3. Browse the internet, interact with web pages, and make HTTP requests
4. Download files from the web
5. Install Python packages
6. Run system commands and manage background processes
7. Inspect system hardware and resources

When given a task, break it down into steps and execute them systematically.
Always verify your work and iterate until the task is completed successfully.

Available tools:
- write_file(filepath, content): Write content to a file
- read_file(filepath): Read content from a file
- append_file(filepath, content): Append to a file
- delete_file(filepath): Delete a file
- list_directory(dirpath): List files and folders in a directory
- move_file(source, destination): Move or rename a file or directory
- copy_file(source, destination): Copy a file or directory
- create_directory(dirpath): Create a directory (and parents)
- remove_directory(dirpath): Remove a directory and all its contents
- execute_code(code, language): Execute code (python/bash/js) and return output
- browse_web(url): Navigate to a webpage and get its content
- search_web(query): Search the web via DuckDuckGo
- interact_with_page(action, selector, value): Interact with the current browser page \
(actions: click, type, extract_text, extract_links, screenshot)
- http_request(url, method, data, headers): Make direct HTTP API calls (GET/POST/PUT/DELETE)
- download_file(url, save_path): Download a file from the internet
- pip_install(packages): Install Python packages into the current environment
- run_command(command): Execute a system command
- spawn_process(name, command): Start a named background process
- check_process(name): Check if a background process is still running
- kill_process(name): Terminate a background process
- system_info(): Get OS, CPU, RAM, GPU, and disk information

Format your responses as JSON with 'action', 'parameters', and 'reasoning' fields.
Set "complete": true when the task is finished.
You may return {"actions": [{...}, {...}]} to execute multiple steps in one turn.
"""


class AutoDeepSeek:
    def __init__(
        self,
        model_path: str = None,
        *,
        full_access: bool = False,
        device_preference: str = "auto",
    ):
        """Initialize AutoDeepSeek agent

        Parameters
        ----------
        model_path : str, optional
            HuggingFace model ID. Defaults to ``MODEL_PATH`` env var or
            ``Qwen/Qwen2.5-Coder-7B-Instruct``.
        full_access : bool, optional
            If True, file and command operations are not restricted to the
            workspace directory.
        device_preference : str
            Preferred compute device: auto, cuda, cpu, or mps.
        """
        if model_path is None:
            model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.model_path = model_path
        self.full_access = full_access
        self.device_preference = device_preference
        self._setup_logging()
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.browser = None
        self.workspace_dir = Path(os.getenv("WORKSPACE_DIR", "./autodeepseek_workspace"))
        self.default_save_dir = Path.home() / "Desktop"
        self.conversation_history: List[Dict] = []
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "25"))
        self.execution_timeout = int(os.getenv("EXECUTION_TIMEOUT", "120"))
        self.abort_flag = False
        self._background_processes: Dict[str, subprocess.Popen] = {}
        self._dml_device = None  # set later if using DirectML

        # Create workspace
        self.workspace_dir.mkdir(exist_ok=True)

        # Initialize components
        self._load_model()
        self._setup_browser()

        self.logger.info("AutoDeepSeek initialized successfully")

    # -- Context Manager -------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    # -- Abort -----------------------------------------------------------------

    def request_abort(self):
        """Signal the agent to abort the current task."""
        self.abort_flag = True

    def reset_abort(self):
        """Clear any abort flags before starting a new task."""
        self.abort_flag = False

    # -- Setup -----------------------------------------------------------------

    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"autodeepseek_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AutoDeepSeek")

    def _setup_device(self) -> str:
        """Determine the best device to run the model on.

        Priority (auto): CUDA → DirectML (Windows) → MPS (macOS) → CPU
        """
        preference = self.device_preference.lower()

        def mps_available() -> bool:
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        def directml_available() -> bool:
            return _DIRECTML_AVAILABLE and platform.system() == "Windows"

        # Explicit preference
        if preference == "cuda" and torch.cuda.is_available():
            self.logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
            return "cuda"
        if preference == "directml" and directml_available():
            self.logger.info(f"Using DirectML (device 0)")
            return "directml"
        if preference == "mps" and mps_available():
            self.logger.info("Using Apple Silicon (MPS)")
            return "mps"
        if preference == "cpu":
            self.logger.info("Using CPU as requested")
            return "cpu"

        # Auto-detection cascade
        if torch.cuda.is_available():
            self.logger.info(f"Auto-selected CUDA GPU: {torch.cuda.get_device_name()}")
            return "cuda"
        if directml_available():
            self.logger.info("Auto-selected DirectML (AMD/Intel GPU via DirectX 12)")
            return "directml"
        if mps_available():
            self.logger.info("Auto-selected Apple Silicon (MPS)")
            return "mps"
        self.logger.warning("No GPU backend available, using CPU")
        return "cpu"

    def _load_model(self):
        """Load model and tokenizer from HuggingFace"""
        try:
            self.logger.info(f"Loading model: {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # DirectML works with float32 or float16; CUDA/MPS use float16
            if self.device == "cpu":
                dtype = torch.float32
            else:
                dtype = torch.float16

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
            }
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            if self.device == "directml":
                dml_device = torch_directml.device()
                self.model = self.model.to(dml_device)
                self._dml_device = dml_device  # store for tensor placement
            elif self.device in ("cpu", "mps"):
                self.model = self.model.to(self.device)

            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _setup_browser(self):
        """Setup Firefox browser with selenium"""
        try:
            options = FirefoxOptions()
            options.add_argument("--width=1920")
            options.add_argument("--height=1080")
            if os.getenv("BROWSER_HEADLESS", "true").lower() != "false":
                options.add_argument("--headless")

            self.browser = webdriver.Firefox(options=options)
            self.logger.info("Browser initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {e}")
            self.browser = None

    # -- LLM Generation --------------------------------------------------------

    def generate_response(self, prompt: str, max_length: int = 2048) -> str:
        """Generate response using the loaded model"""
        try:
            full_prompt = self._build_context_prompt(prompt)

            target_device = self._dml_device if self.device == "directml" else self.device
            inputs = self.tokenizer.encode(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(target_device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=float(os.getenv("TEMPERATURE", "0.7")),
                    do_sample=True,
                    top_p=float(os.getenv("TOP_P", "0.9")),
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )

            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            ).strip()

            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def _build_context_prompt(self, prompt: str) -> str:
        """Build prompt with conversation context and system instructions"""
        context = SYSTEM_PROMPT + "\n\n"
        for entry in self.conversation_history[-5:]:
            context += f"Human: {entry['human']}\nAssistant: {entry['assistant']}\n\n"
        context += f"Human: {prompt}\nAssistant: "
        return context

    # -- Path Resolution & Sandboxing ------------------------------------------

    def _resolve_path(self, filepath: Optional[str], allow_none: bool = False) -> Path:
        """Resolve a filepath respecting workspace sandboxing.

        Raises ValueError if the resolved path escapes the workspace while
        full_access is disabled.
        """
        if filepath is None and allow_none:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.default_save_dir / f"autodeepseek_{timestamp}.txt"
        elif filepath:
            file_path = Path(filepath).expanduser()
        else:
            raise ValueError("filepath is required")

        if not file_path.is_absolute() and not self.full_access:
            file_path = (self.workspace_dir / file_path).resolve()
        elif not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()

        if not self.full_access:
            try:
                file_path.resolve().relative_to(self.workspace_dir.resolve())
            except ValueError:
                raise ValueError(f"Access denied: path '{file_path}' is outside the workspace")

        return file_path

    # -- File Operations -------------------------------------------------------

    def write_file(self, filepath: Optional[str], content: str) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            file_path = self._resolve_path(filepath, allow_none=True)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"File written: {file_path}")
            return {"success": True, "message": f"File written to {file_path}"}
        except Exception as e:
            error_msg = f"Error writing file {filepath}: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def read_file(self, filepath: str) -> Dict[str, Any]:
        """Read content from a file"""
        try:
            file_path = self._resolve_path(filepath)
            if not file_path.exists():
                return {"success": False, "message": f"File {filepath} does not exist"}
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.logger.info(f"File read: {file_path}")
            return {"success": True, "content": content}
        except Exception as e:
            error_msg = f"Error reading file {filepath}: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def append_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """Append content to an existing file"""
        try:
            file_path = self._resolve_path(filepath)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"File appended: {file_path}")
            return {"success": True, "message": f"Content appended to {file_path}"}
        except Exception as e:
            error_msg = f"Error appending file {filepath}: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def delete_file(self, filepath: str) -> Dict[str, Any]:
        """Delete a file"""
        try:
            file_path = self._resolve_path(filepath)
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"File deleted: {file_path}")
                return {"success": True, "message": f"Deleted {file_path}"}
            else:
                return {"success": False, "message": f"File {filepath} does not exist"}
        except Exception as e:
            error_msg = f"Error deleting file {filepath}: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def list_directory(self, dirpath: str = ".") -> Dict[str, Any]:
        """List contents of a directory with file sizes and types."""
        try:
            dir_path = self._resolve_path(dirpath)
            if not dir_path.is_dir():
                return {"success": False, "message": f"Not a directory: {dir_path}"}
            entries = []
            for item in sorted(dir_path.iterdir()):
                entries.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })
            return {"success": True, "path": str(dir_path), "entries": entries}
        except Exception as e:
            return {"success": False, "message": f"Error listing directory: {e}"}

    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move or rename a file or directory."""
        try:
            src = self._resolve_path(source)
            dst = self._resolve_path(destination)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            self.logger.info(f"Moved {src} -> {dst}")
            return {"success": True, "message": f"Moved {src} -> {dst}"}
        except Exception as e:
            return {"success": False, "message": f"Error moving file: {e}"}

    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy a file or directory."""
        try:
            src = self._resolve_path(source)
            dst = self._resolve_path(destination)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            self.logger.info(f"Copied {src} -> {dst}")
            return {"success": True, "message": f"Copied {src} -> {dst}"}
        except Exception as e:
            return {"success": False, "message": f"Error copying file: {e}"}

    def create_directory(self, dirpath: str) -> Dict[str, Any]:
        """Create a directory (and any missing parents)."""
        try:
            dir_path = self._resolve_path(dirpath)
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directory created: {dir_path}")
            return {"success": True, "message": f"Directory created: {dir_path}"}
        except Exception as e:
            return {"success": False, "message": f"Error creating directory: {e}"}

    def remove_directory(self, dirpath: str) -> Dict[str, Any]:
        """Remove a directory and all its contents."""
        try:
            dir_path = self._resolve_path(dirpath)
            if not dir_path.is_dir():
                return {"success": False, "message": f"Not a directory: {dir_path}"}
            shutil.rmtree(dir_path)
            self.logger.info(f"Directory removed: {dir_path}")
            return {"success": True, "message": f"Directory removed: {dir_path}"}
        except Exception as e:
            return {"success": False, "message": f"Error removing directory: {e}"}

    # -- Code Execution --------------------------------------------------------

    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code and return output. Uses temp files to avoid collisions."""
        try:
            suffix_map = {"python": ".py", "bash": ".sh", "javascript": ".js", "js": ".js"}
            suffix = suffix_map.get(language.lower())
            if not suffix:
                return {"success": False, "message": f"Unsupported language: {language}"}

            with tempfile.NamedTemporaryFile(
                mode='w', suffix=suffix, dir=self.workspace_dir,
                delete=False, encoding='utf-8'
            ) as f:
                f.write(code)
                temp_file = Path(f.name)

            if language.lower() == "python":
                command = [sys.executable, str(temp_file)]
            elif language.lower() == "bash":
                os.chmod(temp_file, 0o755)
                command = ["bash", str(temp_file)]
            else:
                command = ["node", str(temp_file)]

            cwd = None if self.full_access else self.workspace_dir
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.execution_timeout,
                cwd=cwd
            )

            result = {
                "success": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "return_code": process.returncode
            }

            temp_file.unlink(missing_ok=True)
            self.logger.info(f"Code executed - Return code: {process.returncode}")
            return result

        except subprocess.TimeoutExpired:
            temp_file.unlink(missing_ok=True)
            error_msg = f"Code execution timed out ({self.execution_timeout}s limit)"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
        except Exception as e:
            error_msg = f"Error executing code: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    # -- Web & HTTP ------------------------------------------------------------

    def browse_web(self, url: str) -> Dict[str, Any]:
        """Navigate to a webpage and return content"""
        if not self.browser:
            return {"success": False, "message": "Browser not available"}
        try:
            self.browser.get(url)
            time.sleep(2)
            title = self.browser.title
            page_source = self.browser.page_source[:5000]
            self.logger.info(f"Browsed to: {url}")
            return {
                "success": True,
                "title": title,
                "content": page_source,
                "url": self.browser.current_url
            }
        except Exception as e:
            error_msg = f"Error browsing {url}: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def search_web(self, query: str) -> Dict[str, Any]:
        """Search the web using DuckDuckGo"""
        try:
            search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
            result = self.browse_web(search_url)
            if result["success"]:
                self.logger.info(f"Web search completed: {query}")
            return result
        except Exception as e:
            return {"success": False, "message": f"Error searching web: {e}"}

    def interact_with_page(self, action: str, selector: str = "", value: str = "") -> Dict[str, Any]:
        """Interact with the current browser page.
        Actions: click, type, extract_text, extract_links, screenshot
        """
        if not self.browser:
            return {"success": False, "message": "Browser not available"}
        try:
            wait = WebDriverWait(self.browser, 10)
            if action == "click":
                elem = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                elem.click()
                return {"success": True, "message": f"Clicked {selector}"}
            elif action == "type":
                elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                elem.clear()
                elem.send_keys(value)
                return {"success": True, "message": f"Typed into {selector}"}
            elif action == "extract_text":
                elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                return {"success": True, "text": elem.text}
            elif action == "extract_links":
                links = self.browser.find_elements(By.TAG_NAME, "a")
                return {"success": True, "links": [
                    {"text": a.text, "href": a.get_attribute("href")} for a in links[:50]
                ]}
            elif action == "screenshot":
                path = self.workspace_dir / "screenshot.png"
                self.browser.save_screenshot(str(path))
                return {"success": True, "message": f"Screenshot saved to {path}"}
            else:
                return {"success": False, "message": f"Unknown page action: {action}"}
        except Exception as e:
            return {"success": False, "message": f"Error interacting with page: {e}"}

    def http_request(self, url: str, method: str = "GET",
                     data: Optional[Dict] = None,
                     headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a direct HTTP request (faster than browser for APIs)."""
        try:
            resp = requests.request(
                method.upper(), url, json=data,
                headers=headers or {}, timeout=30
            )
            try:
                body = resp.json()
            except ValueError:
                body = resp.text[:5000]
            return {
                "success": resp.ok,
                "status_code": resp.status_code,
                "body": body,
            }
        except Exception as e:
            return {"success": False, "message": f"HTTP request failed: {e}"}

    def download_file(self, url: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Download a file from a URL."""
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            if not save_path:
                save_path = url.split("/")[-1].split("?")[0] or "download"
            file_path = self._resolve_path(save_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f"Downloaded {url} -> {file_path}")
            return {"success": True, "message": f"Downloaded to {file_path}",
                    "size": file_path.stat().st_size}
        except Exception as e:
            return {"success": False, "message": f"Download failed: {e}"}

    # -- System Operations -----------------------------------------------------

    def pip_install(self, packages: List[str]) -> Dict[str, Any]:
        """Install Python packages into the current environment."""
        try:
            if isinstance(packages, str):
                packages = [packages]
            cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + packages
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            self.logger.info(f"pip install {' '.join(packages)} -> rc={process.returncode}")
            return {
                "success": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        except Exception as e:
            return {"success": False, "message": f"pip install failed: {e}"}

    def system_info(self) -> Dict[str, Any]:
        """Return system information: OS, CPU, RAM, GPU, disk."""
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage(str(Path.cwd()))
            info = {
                "success": True,
                "os": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "ram_total_gb": round(mem.total / (1024**3), 1),
                "ram_available_gb": round(mem.available / (1024**3), 1),
                "disk_free_gb": round(disk.free / (1024**3), 1),
                "gpu": None,
            }
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
                )
            return info
        except Exception as e:
            return {"success": False, "message": f"Error getting system info: {e}"}

    def run_command(self, command: str) -> Dict[str, Any]:
        """Execute system command safely (no shell injection)."""
        try:
            # Split into args — prevents shell injection
            args = shlex.split(command)
            if not args:
                return {"success": False, "message": "Empty command"}

            # Block dangerous executables
            dangerous = {'rm', 'del', 'format', 'shutdown', 'reboot', 'mkfs', 'fdisk'}
            if args[0].lower() in dangerous:
                return {"success": False, "message": f"Blocked dangerous command: {args[0]}"}

            cwd = None if self.full_access else str(self.workspace_dir)
            process = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.execution_timeout,
                cwd=cwd,
            )
            self.logger.info(f"Command executed: {command}")
            return {
                "success": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "return_code": process.returncode
            }
        except Exception as e:
            error_msg = f"Error running command: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    # -- Background Processes --------------------------------------------------

    def spawn_process(self, name: str, command: str) -> Dict[str, Any]:
        """Start a named background process."""
        try:
            args = shlex.split(command)
            cwd = None if self.full_access else str(self.workspace_dir)
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
            )
            self._background_processes[name] = proc
            self.logger.info(f"Spawned process '{name}' PID={proc.pid}")
            return {"success": True, "message": f"Process '{name}' started (PID {proc.pid})",
                    "pid": proc.pid}
        except Exception as e:
            return {"success": False, "message": f"Error spawning process: {e}"}

    def check_process(self, name: str) -> Dict[str, Any]:
        """Check status of a background process."""
        proc = self._background_processes.get(name)
        if not proc:
            return {"success": False, "message": f"No process named '{name}'"}
        poll = proc.poll()
        return {"success": True, "running": poll is None, "return_code": poll, "pid": proc.pid}

    def kill_process(self, name: str) -> Dict[str, Any]:
        """Terminate a background process."""
        proc = self._background_processes.pop(name, None)
        if not proc:
            return {"success": False, "message": f"No process named '{name}'"}
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        return {"success": True, "message": f"Process '{name}' terminated"}

    # -- Action Dispatch -------------------------------------------------------

    def execute_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action based on the action data"""
        action = action_data.get("action")
        params = action_data.get("parameters", {})

        dispatch = {
            "write_file": lambda: self.write_file(params.get("filepath"), params.get("content")),
            "read_file": lambda: self.read_file(params.get("filepath")),
            "append_file": lambda: self.append_file(params.get("filepath"), params.get("content")),
            "delete_file": lambda: self.delete_file(params.get("filepath")),
            "list_directory": lambda: self.list_directory(params.get("dirpath", ".")),
            "move_file": lambda: self.move_file(params.get("source"), params.get("destination")),
            "copy_file": lambda: self.copy_file(params.get("source"), params.get("destination")),
            "create_directory": lambda: self.create_directory(params.get("dirpath")),
            "remove_directory": lambda: self.remove_directory(params.get("dirpath")),
            "execute_code": lambda: self.execute_code(params.get("code"), params.get("language", "python")),
            "browse_web": lambda: self.browse_web(params.get("url")),
            "search_web": lambda: self.search_web(params.get("query")),
            "interact_with_page": lambda: self.interact_with_page(
                params.get("action"), params.get("selector", ""), params.get("value", "")
            ),
            "http_request": lambda: self.http_request(
                params.get("url"), params.get("method", "GET"),
                params.get("data"), params.get("headers")
            ),
            "download_file": lambda: self.download_file(params.get("url"), params.get("save_path")),
            "pip_install": lambda: self.pip_install(params.get("packages", [])),
            "system_info": lambda: self.system_info(),
            "run_command": lambda: self.run_command(params.get("command")),
            "spawn_process": lambda: self.spawn_process(params.get("name"), params.get("command")),
            "check_process": lambda: self.check_process(params.get("name")),
            "kill_process": lambda: self.kill_process(params.get("name")),
        }

        handler = dispatch.get(action)
        if handler:
            return handler()
        return {"success": False, "message": f"Unknown action: {action}"}

    # -- JSON Parsing ----------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract JSON from a response, stripping markdown fences if present."""
        cleaned = text.strip()
        # Strip ```json ... ``` markdown fences
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
        if cleaned.startswith('{') or cleaned.startswith('['):
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        # Try to find JSON object in the middle of text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    # -- Agentic Task Loop -----------------------------------------------------

    def complete_task(self, task: str) -> str:
        """Complete a complex task autonomously"""
        self.logger.info(f"Starting task: {task}")

        iteration = 0
        task_complete = False
        consecutive_failures = 0
        results = []

        self.reset_abort()
        while iteration < self.max_iterations and not task_complete and not self.abort_flag:
            iteration += 1
            self.logger.info(f"Task iteration {iteration}/{self.max_iterations}")

            # Build prompt
            if iteration == 1:
                prompt = (
                    f"Task: {task}\n\n"
                    "Break this into steps and start executing. "
                    "Respond as JSON with action, parameters, and reasoning. "
                    "Set \"complete\": true when finished."
                )
            elif consecutive_failures > 0 and results:
                last_error = results[-1].get("message", "Unknown error")
                prompt = (
                    f"Your previous action failed with: {last_error}\n"
                    f"Previous results: {json.dumps(results[-3:], indent=2, default=str)}\n\n"
                    "Please fix the issue and try again. Respond as JSON."
                )
            else:
                prompt = (
                    f"Previous results: {json.dumps(results[-3:], indent=2, default=str)}\n\n"
                    "Continue with the next step or set \"complete\": true if done. Respond as JSON."
                )

            response = self.generate_response(prompt)
            action_data = self._extract_json(response)

            if action_data:
                # Handle multi-action batching
                if "actions" in action_data and isinstance(action_data["actions"], list):
                    for sub_action in action_data["actions"]:
                        if self.abort_flag:
                            break
                        result = self.execute_action(sub_action)
                        result["iteration"] = iteration
                        result["reasoning"] = sub_action.get("reasoning", "")
                        results.append(result)
                        print(f"\nIteration {iteration}: {sub_action.get('action')}")
                        print(f"  Reasoning: {result.get('reasoning', '')}")
                        print(f"  Result: {result.get('message', result.get('stdout', 'OK'))}")
                        if not result.get("success", False):
                            consecutive_failures += 1
                            break
                        else:
                            consecutive_failures = 0

                    if action_data.get("complete") and consecutive_failures == 0:
                        task_complete = True

                elif "action" in action_data:
                    result = self.execute_action(action_data)
                    result["iteration"] = iteration
                    result["reasoning"] = action_data.get("reasoning", "")
                    results.append(result)

                    print(f"\nIteration {iteration}:")
                    print(f"  Action: {action_data['action']}")
                    print(f"  Reasoning: {result.get('reasoning', '')}")
                    print(f"  Result: {result.get('message', result.get('stdout', 'OK'))}")

                    if result.get("success", False):
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                    if action_data.get("complete") and result.get("success", False):
                        task_complete = True
                else:
                    # JSON but no action — check for completion flag
                    if action_data.get("complete"):
                        task_complete = True
                    results.append({"iteration": iteration, "response": response, "type": "analysis"})
                    print(f"\nIteration {iteration}: {response[:200]}")
            else:
                # Couldn't parse JSON — store as text, don't treat keywords as completion
                results.append({"iteration": iteration, "response": response, "type": "text"})
                print(f"\nIteration {iteration}: {response[:200]}")

            time.sleep(1)

        # Save conversation history
        self.conversation_history.append({
            "human": task,
            "assistant": f"Task completed in {iteration} iterations",
            "results": results
        })

        if self.abort_flag:
            task_complete = False
            self.logger.info("Task aborted by user")

        # Generate summary
        summary = f"Task: {task}\nCompleted in {iteration} iterations\n"
        if self.abort_flag:
            summary += "Status: ABORTED by user\n"
        elif consecutive_failures > 0 and not task_complete:
            summary += "Status: ERROR - Actions failed\n"
        elif task_complete:
            summary += "Status: SUCCESS - Task completed successfully\n"
        else:
            summary += "Status: INCOMPLETE - Reached maximum iterations\n"

        summary += "\nResults summary:\n"
        for result in results[-3:]:
            if result.get("type") in ("text", "analysis"):
                summary += f"- {result.get('response', '')[:100]}...\n"
            else:
                summary += f"- {result.get('message', 'Action completed')}\n"

        self.logger.info(f"Task completed: {task}")
        return summary

    # -- Cleanup ---------------------------------------------------------------

    def cleanup(self):
        """Clean up resources"""
        for name, proc in self._background_processes.items():
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._background_processes.clear()

        if self.browser:
            try:
                self.browser.quit()
            except Exception:
                pass
        self.logger.info("AutoDeepSeek cleanup completed")

    def __del__(self):
        """Fallback destructor"""
        try:
            self.cleanup()
        except Exception:
            pass


def main():
    """Main function for command line usage"""
    print("🚀 AutoDeepSeek - Autonomous AI Agent")
    print("=====================================")

    import argparse

    parser = argparse.ArgumentParser(description="Run AutoDeepSeek")
    parser.add_argument("--full-access", action="store_true",
                        help="Allow file and command operations outside the workspace")
    parser.add_argument("--model",
                        default=os.getenv("MODEL_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct"),
                        help="Hugging Face model ID to load")
    parser.add_argument("--device", choices=["auto", "cuda", "directml", "cpu", "mps"],
                        default=os.getenv("DEVICE", "auto"),
                        help="Preferred device backend (directml for AMD GPUs on Windows)")
    args = parser.parse_args()

    full_access_env = os.getenv("FULL_ACCESS", "false").lower() == "true"

    try:
        with AutoDeepSeek(
            model_path=args.model,
            full_access=args.full_access or full_access_env,
            device_preference=args.device,
        ) as agent:
            print("\nAgent initialized successfully!")
            print("Type 'exit' to quit, 'help' for commands")

            while True:
                try:
                    task = input("\n🎯 Enter task: ").strip()

                    if task.lower() == 'exit':
                        break
                    elif task.lower() == 'help':
                        print("""
Available commands:
- Any task description (e.g., "create a Python web scraper")
- 'workspace' - show workspace directory
- 'logs' - show recent logs
- 'clear' - clear conversation history
- 'sysinfo' - show system information
- 'exit' - quit the program
                        """)
                        continue
                    elif task.lower() == 'workspace':
                        print(f"Workspace: {agent.workspace_dir.absolute()}")
                        continue
                    elif task.lower() == 'clear':
                        agent.conversation_history.clear()
                        print("Conversation history cleared")
                        continue
                    elif task.lower() == 'sysinfo':
                        info = agent.system_info()
                        for k, v in info.items():
                            if k != "success":
                                print(f"  {k}: {v}")
                        continue
                    elif task.lower() == 'logs':
                        try:
                            log_files = list(Path("./logs").glob("*.log"))
                            if log_files:
                                latest_log = max(log_files, key=os.path.getmtime)
                                with open(latest_log, 'r') as f:
                                    lines = f.readlines()
                                    print("\nRecent logs:")
                                    for line in lines[-10:]:
                                        print(line.strip())
                            else:
                                print("No log files found")
                        except Exception as e:
                            print(f"Error reading logs: {e}")
                        continue

                    if not task:
                        continue

                    print(f"\n🔄 Processing task: {task}")
                    result = agent.complete_task(task)
                    print(f"\n✅ Task Result:\n{result}")

                except KeyboardInterrupt:
                    print("\n\nTask interrupted by user")
                    continue
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    continue

    except Exception as e:
        print(f"❌ Failed to initialize AutoDeepSeek: {e}")
        return 1

    finally:
        print("\n👋 Goodbye!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
