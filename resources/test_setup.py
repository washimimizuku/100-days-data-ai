"""
Setup verification script for 100 Days of Data and AI
Run this to verify your environment is ready
"""

import sys
import subprocess

def check_python_version():
    """Check Python version is 3.11+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro}")
        print("   Please install Python 3.11 or higher")
        return False

def check_package(package_name, display_name=None):
    """Check if a package is installed"""
    display = display_name or package_name
    try:
        __import__(package_name)
        print(f"✅ {display} installed")
        return True
    except ImportError:
        print(f"❌ {display} not installed")
        print(f"   Run: pip install {package_name}")
        return False

def check_command(command, display_name):
    """Check if a command-line tool is available"""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ {display_name} installed")
            return True
        else:
            print(f"❌ {display_name} not found")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"❌ {display_name} not found")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("Checking setup for 100 Days of Data and AI")
    print("="*60 + "\n")
    
    # Python version
    print("Python Environment:")
    python_ok = check_python_version()
    print()
    
    # Core Python packages (Phase 1)
    print("Core Python Packages (Required for Phase 1):")
    core_checks = [
        check_package("numpy"),
        check_package("pandas"),
        check_package("matplotlib"),
        check_package("seaborn"),
        check_package("jupyter"),
        check_package("duckdb"),
    ]
    print()
    
    # Data Engineering packages (Phase 1)
    print("Data Engineering Packages (Install when needed):")
    de_checks = [
        check_package("pyarrow", "Apache Arrow (pyarrow)"),
        check_package("fastparquet"),
        check_package("pyspark", "PySpark"),
        check_package("kafka", "Kafka Python"),
    ]
    print()
    
    # ML/AI packages (Phase 3-4)
    print("ML/AI Packages (Install for Phase 3-4):")
    ml_checks = [
        check_package("sklearn", "scikit-learn"),
        check_package("torch", "PyTorch"),
        check_package("transformers", "Hugging Face Transformers"),
        check_package("langchain"),
        check_package("chromadb", "ChromaDB"),
    ]
    print()
    
    # API & Development packages (Phase 2)
    print("Development Packages (Install for Phase 2):")
    dev_checks = [
        check_package("fastapi"),
        check_package("pydantic"),
        check_package("pytest"),
        check_package("sqlalchemy"),
    ]
    print()
    
    # Command-line tools
    print("Command-Line Tools:")
    tool_checks = [
        check_command("docker", "Docker"),
        check_command("git", "Git"),
    ]
    print()
    
    # Summary
    print("="*60)
    if python_ok and all(core_checks):
        print("✅ Core setup complete! Ready to start Phase 1")
        print()
        print("📝 Notes:")
        print("   • Install other packages as you progress through phases")
        print("   • Docker is required for Days 44-49")
        print("   • AWS CLI optional for Days 89-90")
        print("   • 16GB+ RAM recommended for Spark and ML work")
        print()
        print("🚀 Start with: Day 1 - CSV vs JSON")
    else:
        print("❌ Setup incomplete. Please install missing requirements:")
        print()
        if not python_ok:
            print("   1. Install Python 3.11+")
        if not all(core_checks):
            print("   2. Install core packages:")
            print("      pip install -r requirements.txt")
        if not all(tool_checks):
            print("   3. Install Docker and Git")
    print("="*60)
    
    return 0 if (python_ok and all(core_checks)) else 1

if __name__ == "__main__":
    sys.exit(main())
