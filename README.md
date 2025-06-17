# 🚀 AutoDeepSeek

An autonomous AI agent powered by DeepSeek-R1 7B with full system access capabilities, designed for Linux systems with AMD GPU (ROCm) support.

## ✨ Features

- **Autonomous Task Completion**: Give it a high-level task and watch it break it down and execute
- **Code Generation & Execution**: Write, save, run, and verify code in multiple languages
- **Web Browsing**: Search the web and navigate websites using Firefox
- **File Management**: Create, read, modify any files in the workspace
- **System Commands**: Execute system commands with safety restrictions
- **GPU Acceleration**: Optimized for AMD GPUs using ROCm on Linux
- **Iterative Problem Solving**: Automatically refines solutions until success
- **Comprehensive Logging**: Full activity logging and conversation history

## 🏗️ Architecture

```
AutoDeepSeek Agent
├── DeepSeek-R1 7B Model (Core Intelligence)
├── Browser Automation (Selenium + Firefox)
├── Code Execution Engine (Python, Bash, JavaScript)
├── File System Manager
├── System Command Interface
├── Task Orchestrator
└── Safety & Security Layer
```

## 📋 Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Hardware**: AMD GPU (RX 6800 XT or better) for optimal performance
- **Python**: 3.8 or higher
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 50GB free space for models and workspace

## 🛠️ Installation

### Automated Setup (Recommended)

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/autodeepseek.git
cd autodeepseek
```

2. **Run the automated setup**:
```bash
python3 setup.py
```

3. **Reboot your system** (required after ROCm installation):
```bash
sudo reboot
```

4. **Test the installation**:
```bash
./run_autodeepseek.sh
python3 test_autodeepseek.py
```

### Manual Setup

If you prefer manual installation:

#### 1. Install ROCm for AMD GPU

```bash
# Add ROCm repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm
sudo apt update
sudo apt install -y rocm-dkms rocm-libs rocm-dev rocm-utils

# Add user to groups
sudo usermod -a -G render,video $USER

# Reboot required
sudo reboot
```

#### 2. Install Firefox and Geckodriver

```bash
# Install Firefox
sudo apt install -y firefox

# Download and install geckodriver
wget -O /tmp/geckodriver.tar.gz https://github.com/mozilla/geckodriver/releases/latest/download/geckodriver-v0.33.0-linux64.tar.gz
cd /tmp && tar -xzf geckodriver.tar.gz
sudo mv geckodriver /usr/local/bin/
sudo chmod +x /usr/local/bin/geckodriver
```

#### 3. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Install other dependencies
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
# Copy configuration template
cp .env.example .env

# Edit configuration as needed
nano .env
```

## 🚀 Usage

### Command Line Interface

Start the interactive agent:

```bash
./run_autodeepseek.sh
```

### Example Tasks

Here are some example tasks you can give to AutoDeepSeek:

#### 1. **Coding Tasks**
```
Create a Python web scraper that extracts news headlines from a website, saves them to a CSV file, and generates a simple HTML report
```

#### 2. **Data Analysis**
```
Analyze the sales data in data.csv, create visualizations showing trends, and generate a summary report with insights
```

#### 3. **Web Development**
```
Build a simple REST API server using Flask that handles user registration and login, with SQLite database storage
```

#### 4. **System Administration**
```
Create a system monitoring script that checks disk usage, memory, and CPU, then generates alerts if thresholds are exceeded
```

#### 5. **Research Tasks**
```
Research the latest developments in quantum computing, summarize the key findings, and create a presentation outline
```

### Programmatic Usage

You can also use AutoDeepSeek programmatically:

```python
from autodeepseek import AutoDeepSeek

# Initialize agent
agent = AutoDeepSeek()

# Execute a task
result = agent.complete_task("Create a calculator program in Python with GUI")
print(result)

# Individual operations
agent.write_file("test.py", "print('Hello World')")
output = agent.execute_code("print('Hello World')", "python")
web_result = agent.browse_web("https://example.com")

# Cleanup
agent.cleanup()
```

## ⚙️ Configuration

Edit the `.env` file to customize AutoDeepSeek:

```bash
# Model Configuration
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DEVICE=auto
MAX_ITERATIONS=10

# Performance Settings
LOW_CPU_MEM_USAGE=true
LOAD_IN_8BIT=false

# Security Settings
ALLOW_SYSTEM_COMMANDS=true
BLOCKED_COMMANDS=rm -rf,format,shutdown
```

## 🔒 Security Features

AutoDeepSeek includes several security measures:

- **Command Filtering**: Dangerous system commands are blocked
- **Workspace Isolation**: All file operations are contained within the workspace
- **Code Sandboxing**: Code execution is limited by timeouts and resource constraints
- **User Confirmation**: Critical operations can require user approval
- **Audit Logging**: All actions are logged for review

## 📊 Performance Optimization

### For AMD GPUs:
```bash
# Set optimal environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
```

### Model Optimization:
- Enable 8-bit quantization: `LOAD_IN_8BIT=true`
- Use model compilation: `TORCH_COMPILE=true`
- Adjust batch size and context length based on available VRAM

## 🐛 Troubleshooting

### Common Issues

#### 1. **ROCm Not Detected**
```bash
# Check ROCm installation
rocm-smi
echo $ROCM_PATH

# Verify user groups
groups $USER
```

#### 2. **Model Loading Errors**
```bash
# Clear model cache
rm -rf ./models_cache

# Check available memory
free -h
nvidia-smi  # or rocm-smi
```

#### 3. **Browser Issues**
```bash
# Check Firefox and geckodriver
firefox --version
geckodriver --version

# Install missing dependencies
sudo apt install -y firefox-esr
```

#### 4. **Permission Errors**
```bash
# Fix workspace permissions
chmod -R 755 autodeepseek_workspace

# Check user groups for GPU access
sudo usermod -a -G render,video $USER
```

### Performance Issues

1. **Slow Model Loading**:
   - Use model quantization
   - Increase system RAM
   - Use SSD storage for model cache

2. **GPU Memory Errors**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller model variants

3. **Browser Timeouts**:
   - Increase timeout values in config
   - Check internet connection
   - Use headless mode for better performance

## 📝 Logging and Debugging

AutoDeepSeek provides comprehensive logging:

```bash
# View recent logs
tail -f logs/autodeepseek_*.log

# Debug mode
LOG_LEVEL=DEBUG ./run_autodeepseek.sh

# Test specific components
python3 test_autodeepseek.py
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python3 test_autodeepseek.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek** for the excellent R1 model
- **AMD** for ROCm GPU computing platform
- **Mozilla** for Firefox and geckodriver
- **Hugging Face** for the transformers library

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/autodeepseek/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/autodeepseek/discussions)
- **Email**: support@autodeepseek.com

---

## 🎯 Example Session

```bash
$ ./run_autodeepseek.sh

🚀 AutoDeepSeek - Autonomous AI Agent
=====================================

Agent initialized successfully!
Type 'exit' to quit, 'help' for commands

🎯 Enter task: Create a web scraper that gets the latest Python news from Python.org and saves it to a JSON file

🔄 Processing task: Create a web scraper that gets the latest Python news from Python.org and saves it to a JSON file

Iteration 1:
Action: browse_web
Reasoning: First, I need to visit Python.org to understand the structure of their news section
Result: Successfully browsed to https://www.python.org/jobs/

Iteration 2:
Action: write_file
Reasoning: Creating a Python web scraper using requests and BeautifulSoup
Result: File written to python_news_scraper.py

Iteration 3:
Action: execute_code
Reasoning: Running the scraper to test if it works correctly
Result: Scraper executed successfully, found 10 news articles

Iteration 4:
Action: read_file
Reasoning: Verifying the JSON output file was created correctly
Result: JSON file contains properly formatted news data

✅ Task Result:
Task: Create a web scraper that gets the latest Python news from Python.org and saves it to a JSON file
Completed in 4 iterations
Status: SUCCESS - Task completed successfully

Results summary:
- Successfully created python_news_scraper.py
- Scraped 10 latest news articles from Python.org
- Saved data to python_news.json with proper formatting
```

Ready to give your AutoDeepSeek agent a task? 🚀