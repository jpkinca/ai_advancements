#!/usr/bin/env python3
"""
AI Trading Advancements Repository Initialization Script

This script sets up the development environment for the AI Trading Advancements project.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"[RUNNING] {description}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"[SUCCESS] {description}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"[ERROR] Python 3.9+ required, found {version.major}.{version.minor}.{version.micro}")
        return False

def setup_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("[CREATING] Virtual environment...")
        run_command("python -m venv venv", "Creating virtual environment")
    else:
        print("[EXISTS] Virtual environment found")
    
    # Activation command varies by OS
    if os.name == 'nt':  # Windows
        activate_cmd = r"venv\Scripts\activate"
        pip_cmd = r"venv\Scripts\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    return pip_cmd

def install_dependencies(pip_cmd):
    """Install required dependencies."""
    print("[INSTALLING] Dependencies...")
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if Path("requirements.txt").exists():
        run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements")
    
    # Install development dependencies
    dev_packages = [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0", 
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.4.0",
        "pre-commit>=3.3.0"
    ]
    
    for package in dev_packages:
        run_command(f"{pip_cmd} install {package}", f"Installing {package}")

def setup_git_hooks():
    """Set up pre-commit hooks."""
    print("[SETTING UP] Git hooks...")
    
    # Create pre-commit config if it doesn't exist
    precommit_config = """repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
        language_version: python3
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
"""
    
    if not Path(".pre-commit-config.yaml").exists():
        with open(".pre-commit-config.yaml", "w") as f:
            f.write(precommit_config)
        print("[CREATED] Pre-commit configuration")
    
    run_command("pre-commit install", "Installing pre-commit hooks")

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    if not Path(".env").exists() and Path(".env.example").exists():
        print("[CREATING] Environment file from template...")
        run_command("cp .env.example .env", "Creating .env file")
        print("[NOTICE] Please edit .env file with your configuration")
    elif Path(".env").exists():
        print("[EXISTS] Environment file found")
    else:
        print("[WARNING] No .env.example template found")

def verify_installation():
    """Verify that the installation was successful."""
    print("\n[VERIFYING] Installation...")
    
    # Check if key packages are importable
    test_imports = [
        "numpy",
        "pandas", 
        "sklearn",
        "faiss",
        "psycopg2",
        "sqlalchemy"
    ]
    
    for package in test_imports:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[WARNING] {package} not available")

def main():
    """Main initialization function."""
    print("AI Trading Advancements - Repository Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    pip_cmd = setup_virtual_environment()
    
    # Install dependencies
    install_dependencies(pip_cmd)
    
    # Setup git hooks
    setup_git_hooks()
    
    # Create environment file
    create_env_file()
    
    # Verify installation
    verify_installation()
    
    print("\n[COMPLETED] Repository setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your configuration")
    print("2. Review README.md for detailed usage instructions")
    print("3. Run tests: pytest")
    print("4. Start development!")

if __name__ == "__main__":
    main()
