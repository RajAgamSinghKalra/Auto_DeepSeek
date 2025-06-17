#!/usr/bin/env python3
"""
AutoDeepSeek Test Script
Tests all components and capabilities
"""

import sys
import os
import json
import time
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from autodeepseek import AutoDeepSeek
except ImportError as e:
    print(f"❌ Cannot import AutoDeepSeek: {e}")
    print("Make sure you've run setup.py and activated the virtual environment")
    sys.exit(1)

def test_basic_functionality():
    """Test basic AutoDeepSeek functionality"""
    print("🧪 Testing Basic Functionality")
    print("="*40)
    
    try:
        # Initialize agent
        print("1. Initializing AutoDeepSeek...")
        agent = AutoDeepSeek()
        print("✅ AutoDeepSeek initialized successfully")
        
        # Test file operations
        print("\n2. Testing file operations...")
        test_content = "# Test File\nprint('Hello from AutoDeepSeek!')\n"
        result = agent.write_file("test.py", test_content)
        
        if result["success"]:
            print("✅ File write successful")
            
            # Read the file back
            read_result = agent.read_file("test.py")
            if read_result["success"] and read_result["content"] == test_content:
                print("✅ File read successful")
            else:
                print("❌ File read failed")
                return False
        else:
            print("❌ File write failed")
            return False
        
        # Test code execution
        print("\n3. Testing code execution...")
        code_result = agent.execute_code("print('Hello from executed code!')", "python")
        
        if code_result["success"]:
            print("✅ Code execution successful")
            print(f"Output: {code_result['stdout'].strip()}")
        else:
            print("❌ Code execution failed")
            return False
        
        # Test web browsing (if browser available)
        print("\n4. Testing web browsing...")
        if agent.browser:
            browse_result = agent.browse_web("https://httpbin.org/json")
            if browse_result["success"]:
                print("✅ Web browsing successful")
            else:
                print("⚠️  Web browsing failed (may be network related)")
        else:
            print("⚠️  Browser not available")
        
        # Test model response
        print("\n5. Testing model response...")
        response = agent.generate_response("What is 2+2? Respond with just the number.")
        if response:
            print(f"✅ Model response: {response}")
        else:
            print("❌ Model response failed")
            return False
        
        print("\n✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        try:
            agent.cleanup()
        except:
            pass

def test_task_completion():
    """Test autonomous task completion"""
    print("\n🎯 Testing Task Completion")
    print("="*40)
    
    try:
        agent = AutoDeepSeek()
        
        # Simple coding task
        task = """Create a Python function that calculates the factorial of a number, 
                 save it to a file called 'factorial.py', and test it with the number 5."""
        
        print(f"Task: {task}")
        print("\nExecuting task...")
        
        result = agent.complete_task(task)
        print(f"\nTask Result:\n{result}")
        
        # Check if factorial.py was created
        factorial_file = Path("autodeepseek_workspace/factorial.py")
        if factorial_file.exists():
            print("\n✅ Task completed successfully - factorial.py created")
            with open(factorial_file, 'r') as f:
                print(f"File contents:\n{f.read()}")
            return True
        else:
            print("\n⚠️  Task may not have completed fully")
            return False
        
    except Exception as e:
        print(f"❌ Task completion test failed: {e}")
        return False
    finally:
        try:
            agent.cleanup()
        except:
            pass

def test_system_info():
    """Display system information"""
    print("\n💻 System Information")
    print("="*40)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Check ROCm
        try:
            import subprocess
            result = subprocess.run("rocm-smi", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ ROCm available")
            else:
                print("⚠️  ROCm not available")
        except:
            print("⚠️  ROCm not available")
        
        # Check transformers
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        # Check selenium
        from selenium import webdriver
        print("✅ Selenium available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking system info: {e}")
        return False

def run_performance_test():
    """Run performance benchmarks"""
    print("\n⚡ Performance Test")
    print("="*40)
    
    try:
        agent = AutoDeepSeek()
        
        # Test model loading time
        start_time = time.time()
        response = agent.generate_response("Hello, how are you?")
        end_time = time.time()
        
        print(f"Model inference time: {end_time - start_time:.2f} seconds")
        print(f"Response: {response[:100]}...")
        
        # Test file operations performance
        start_time = time.time()
        for i in range(10):
            agent.write_file(f"test_{i}.txt", f"Test content {i}")
        end_time = time.time()
        
        print(f"File operations (10 files): {end_time - start_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    finally:
        try:
            agent.cleanup()
        except:
            pass

def main():
    """Main test function"""
    print("🧪 AutoDeepSeek Test Suite")
    print("="*50)
    
    tests = [
        ("System Information", test_system_info),
        ("Basic Functionality", test_basic_functionality),
        ("Task Completion", test_task_completion),
        ("Performance", run_performance_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Results Summary")
    print('='*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AutoDeepSeek is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())