#!/usr/bin/env python3
"""
AutoDeepSeek - Autonomous AI Agent using DeepSeek-R1 7B
Main agent orchestrator with full system access capabilities
"""

import os
import sys
import json
import time
import subprocess
import threading
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

class AutoDeepSeek:
    def __init__(self, model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", *, full_access: bool = False):
        """Initialize AutoDeepSeek agent

        Parameters
        ----------
        model_path: str
            HuggingFace path of the DeepSeek model.
        full_access: bool, optional
            If True, file and command operations are not restricted to the
            workspace directory. When enabled, files will be saved to the
            user's Desktop if no path is provided.
        """
        self.model_path = model_path
        self.full_access = full_access
        self._setup_logging()
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.browser = None
        self.workspace_dir = Path("./autodeepseek_workspace")
        self.default_save_dir = Path.home() / "Desktop"
        self.conversation_history = []
        self.max_iterations = 10
        self.abort_flag = False
        
        
        # Create workspace
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._load_model()
        self._setup_browser()

        self.logger.info("AutoDeepSeek initialized successfully")

    def request_abort(self):
        """Signal the agent to abort the current task."""
        self.abort_flag = True

    def reset_abort(self):
        """Clear any abort flags before starting a new task."""
        self.abort_flag = False

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
        """Determine the best device to run the model on."""
        import platform

        if torch.cuda.is_available():
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            return "cuda"

        if platform.system() == "Windows":
            self.logger.info("No compatible GPU detected on Windows, using CPU")
            return "cpu"

        self.logger.warning("GPU not available, using CPU")
        return "cpu"

    def _load_model(self):
        """Load DeepSeek model and tokenizer"""
        try:
            self.logger.info(f"Loading model: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings for ROCm/CUDA
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            if self.device == "cpu":
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
            # Remove headless for debugging, add back if needed
            # options.add_argument("--headless")
            
            self.browser = webdriver.Firefox(options=options)
            self.logger.info("Browser initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {e}")
            self.browser = None

    def generate_response(self, prompt: str, max_length: int = 2048) -> str:
        """Generate response using DeepSeek model"""
        try:
            # Add conversation context
            full_prompt = self._build_context_prompt(prompt)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
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
        system_prompt = """You are AutoDeepSeek, an autonomous AI agent with full system access. You can:
1. Write and execute code in any programming language
2. Create, read, modify or delete files
3. Browse the internet using Firefox
4. Run system commands
5. Analyze and verify your work

When given a task, break it down into steps and execute them systematically.
Always verify your work and iterate until the task is completed successfully.

Available tools:
- write_file(filepath, content): Write content to a file. When no path is provided and full access is enabled, files are saved to the user's Desktop.
- read_file(filepath): Read content from a file
- append_file(filepath, content): Append text to a file
- delete_file(filepath): Delete a file
- execute_code(code, language): Execute code and return output
- browse_web(url): Navigate to a webpage
- search_web(query): Search the web
- run_command(command): Execute system command

Format your responses as JSON with 'action', 'parameters', and 'reasoning' fields when taking actions.
"""
        
        context = system_prompt + "\n\n"
        
        # Add recent conversation history
        for entry in self.conversation_history[-5:]:  # Last 5 exchanges
            context += f"Human: {entry['human']}\nAssistant: {entry['assistant']}\n\n"
        
        context += f"Human: {prompt}\nAssistant: "
        
        return context

    def write_file(self, filepath: Optional[str], content: str) -> Dict[str, Any]:
        """Write content to a file. If no path is provided and full access is
        enabled, the file is saved to the user's Desktop."""
        try:
            if filepath:
                file_path = Path(filepath).expanduser()
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = self.default_save_dir / f"autodeepseek_{timestamp}.txt"

            if not file_path.is_absolute() and not self.full_access:
                file_path = (self.workspace_dir / file_path).resolve()
            elif not file_path.is_absolute():
                file_path = (Path.cwd() / file_path).resolve()

            if not self.full_access and not str(file_path).startswith(str(self.workspace_dir.resolve())):
                raise ValueError("Attempted write outside workspace")

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
            file_path = Path(filepath).expanduser()
            if not file_path.is_absolute() and not self.full_access:
                file_path = (self.workspace_dir / file_path).resolve()
            elif not file_path.is_absolute():
                file_path = (Path.cwd() / file_path).resolve()

            if not self.full_access and not str(file_path).startswith(str(self.workspace_dir.resolve())):
                raise ValueError("Attempted read outside workspace")

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
            file_path = Path(filepath).expanduser()
            if not file_path.is_absolute() and not self.full_access:
                file_path = (self.workspace_dir / file_path).resolve()
            elif not file_path.is_absolute():
                file_path = (Path.cwd() / file_path).resolve()

            if not self.full_access and not str(file_path).startswith(str(self.workspace_dir.resolve())):
                raise ValueError("Attempted write outside workspace")

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
            file_path = Path(filepath).expanduser()
            if not file_path.is_absolute() and not self.full_access:
                file_path = (self.workspace_dir / file_path).resolve()
            elif not file_path.is_absolute():
                file_path = (Path.cwd() / file_path).resolve()

            if not self.full_access and not str(file_path).startswith(str(self.workspace_dir.resolve())):
                raise ValueError("Attempted delete outside workspace")

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

    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code and return output"""
        try:
            # Create temporary file
            if language.lower() == "python":
                temp_file = self.workspace_dir / "temp_code.py"
                command = [sys.executable, str(temp_file)]
            elif language.lower() == "bash":
                temp_file = self.workspace_dir / "temp_code.sh"
                command = ["bash", str(temp_file)]
            elif language.lower() in ["javascript", "js"]:
                temp_file = self.workspace_dir / "temp_code.js"
                command = ["node", str(temp_file)]
            else:
                return {"success": False, "message": f"Unsupported language: {language}"}
            
            # Write code to temp file
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            if language.lower() == "bash":
                os.chmod(temp_file, 0o755)
            
            # Execute code
            cwd = None if self.full_access else self.workspace_dir
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwd
            )
            
            result = {
                "success": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "return_code": process.returncode
            }
            
            # Clean up temp file
            temp_file.unlink()
            
            self.logger.info(f"Code executed - Return code: {process.returncode}")
            return result
            
        except subprocess.TimeoutExpired:
            error_msg = "Code execution timed out (30s limit)"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
        except Exception as e:
            error_msg = f"Error executing code: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def browse_web(self, url: str) -> Dict[str, Any]:
        """Navigate to a webpage and return content"""
        if not self.browser:
            return {"success": False, "message": "Browser not available"}
        
        try:
            self.browser.get(url)
            time.sleep(2)  # Wait for page load
            
            title = self.browser.title
            page_source = self.browser.page_source[:5000]  # First 5000 chars
            
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
            error_msg = f"Error searching web: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def run_command(self, command: str) -> Dict[str, Any]:
        """Execute system command"""
        try:
            # Security check - block dangerous commands
            dangerous_commands = ['rm -rf', 'format', 'del /f', 'shutdown', 'reboot']
            if any(dangerous in command.lower() for dangerous in dangerous_commands):
                return {"success": False, "message": "Dangerous command blocked"}

            cwd = None if self.full_access else self.workspace_dir
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwd
            )
            
            result = {
                "success": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "return_code": process.returncode
            }
            
            self.logger.info(f"Command executed: {command}")
            return result
            
        except Exception as e:
            error_msg = f"Error running command: {e}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def execute_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action based on the action data"""
        action = action_data.get("action")
        params = action_data.get("parameters", {})
        
        if action == "write_file":
            return self.write_file(params.get("filepath"), params.get("content"))
        elif action == "read_file":
            return self.read_file(params.get("filepath"))
        elif action == "append_file":
            return self.append_file(params.get("filepath"), params.get("content"))
        elif action == "delete_file":
            return self.delete_file(params.get("filepath"))
        elif action == "execute_code":
            return self.execute_code(params.get("code"), params.get("language", "python"))
        elif action == "browse_web":
            return self.browse_web(params.get("url"))
        elif action == "search_web":
            return self.search_web(params.get("query"))
        elif action == "run_command":
            return self.run_command(params.get("command"))
        else:
            return {"success": False, "message": f"Unknown action: {action}"}

    def complete_task(self, task: str) -> str:
        """Complete a complex task autonomously"""
        self.logger.info(f"Starting task: {task}")
        
        iteration = 0
        task_complete = False
        action_failed = False
        results = []
        
        self.reset_abort()
        while iteration < self.max_iterations and not task_complete and not self.abort_flag:
            iteration += 1
            self.logger.info(f"Task iteration {iteration}/{self.max_iterations}")
            
            # Generate next step
            if iteration == 1:
                prompt = f"Task: {task}\n\nPlease break this down into steps and start executing. Provide your response as JSON with action, parameters, and reasoning."
            else:
                prompt = f"Previous results: {json.dumps(results[-3:], indent=2)}\n\nContinue with the next step or verify if the task is complete. Provide JSON response."
            
            response = self.generate_response(prompt)
            
            try:
                # Try to parse JSON response
                if response.strip().startswith('{'):
                    action_data = json.loads(response)
                    
                    if "action" in action_data:
                        # Execute the action
                        result = self.execute_action(action_data)
                        result["iteration"] = iteration
                        result["reasoning"] = action_data.get("reasoning", "")
                        results.append(result)
                        if not result.get("success", False):
                            action_failed = True

                        print(f"\nIteration {iteration}:")
                        print(f"Action: {action_data['action']}")
                        print(f"Reasoning: {result['reasoning']}")
                        print(f"Result: {result.get('message', result.get('stdout', 'Success'))}")

                        # Check if task is marked as complete
                        if (
                            "complete" in action_data
                            and action_data["complete"]
                            and result.get("success", False)
                        ):
                            task_complete = True
                    else:
                        # Regular response, check for completion indicators
                        if any(word in response.lower() for word in ["complete", "finished", "done", "success"]):
                            task_complete = True
                        
                        results.append({
                            "iteration": iteration,
                            "response": response,
                            "type": "analysis"
                        })
                        
                        print(f"\nIteration {iteration}: {response}")
                
                else:
                    # Regular text response
                    results.append({
                        "iteration": iteration,
                        "response": response,
                        "type": "text"
                    })
                    
                    print(f"\nIteration {iteration}: {response}")
                    
                    # Check for completion indicators
                    if any(word in response.lower() for word in ["complete", "finished", "done", "task completed"]):
                        task_complete = True
                
            except json.JSONDecodeError:
                # Handle non-JSON responses
                results.append({
                    "iteration": iteration,
                    "response": response,
                    "type": "text"
                })
                
                print(f"\nIteration {iteration}: {response}")
            
            time.sleep(1)  # Brief pause between iterations
        
        # Save conversation history
        self.conversation_history.append({
            "human": task,
            "assistant": f"Task completed in {iteration} iterations",
            "results": results
        })

        if self.abort_flag:
            task_complete = False
            self.logger.info("Task aborted by user")
        if action_failed and task_complete:
            task_complete = False
        
        # Generate summary
        summary = f"Task: {task}\nCompleted in {iteration} iterations\n"
        if self.abort_flag:
            summary += "Status: ABORTED by user\n"
        elif action_failed:
            summary += "Status: ERROR - Some actions failed\n"
        elif task_complete:
            summary += "Status: SUCCESS - Task completed successfully\n"
        else:
            summary += "Status: INCOMPLETE - Reached maximum iterations\n"
        
        summary += f"\nResults summary:\n"
        for result in results[-3:]:  # Last 3 results
            if result.get("type") == "text":
                summary += f"- {result['response'][:100]}...\n"
            else:
                summary += f"- {result.get('message', 'Action completed')}\n"
        
        self.logger.info(f"Task completed: {task}")
        return summary

    def cleanup(self):
        """Clean up resources"""
        if self.browser:
            self.browser.quit()
        self.logger.info("AutoDeepSeek cleanup completed")

    def __del__(self):
        """Destructor"""
        self.cleanup()


def main():
    """Main function for command line usage"""
    print("🚀 AutoDeepSeek - Autonomous AI Agent")
    print("=====================================")

    import argparse

    parser = argparse.ArgumentParser(description="Run AutoDeepSeek")
    parser.add_argument("--full-access", action="store_true", help="Allow file and command operations outside the workspace")
    args = parser.parse_args()

    full_access_env = os.getenv("FULL_ACCESS", "false").lower() == "true"

    try:
        # Initialize agent
        agent = AutoDeepSeek(full_access=args.full_access or full_access_env)
        
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
                elif task.lower() == 'logs':
                    try:
                        log_files = list(Path("./logs").glob("*.log"))
                        if log_files:
                            latest_log = max(log_files, key=os.path.getmtime)
                            with open(latest_log, 'r') as f:
                                lines = f.readlines()
                                print("\nRecent logs:")
                                for line in lines[-10:]:  # Last 10 lines
                                    print(line.strip())
                        else:
                            print("No log files found")
                    except Exception as e:
                        print(f"Error reading logs: {e}")
                    continue
                
                if not task:
                    continue
                
                # Execute task
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
