#!/usr/bin/env python3
"""
Setup script for Real-Time VAD development environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {command}")
    if isinstance(command, str):
        command = command.split()
    
    result = subprocess.run(
        command,
        check=check,
        capture_output=capture_output,
        text=True
    )
    
    if capture_output:
        return result.stdout.strip()
    return result


def check_python_version():
    """Check if Python version is supported."""
    version = sys.version_info
    if version < (3, 8):
        print(f"Error: Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        sys.exit(1)
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is supported")


def create_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return venv_path
    
    print("Creating virtual environment...")
    run_command([sys.executable, "-m", "venv", "venv"])
    print("✓ Virtual environment created")
    
    return venv_path


def get_venv_python(venv_path):
    """Get the path to Python executable in virtual environment."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def get_venv_pip(venv_path):
    """Get the path to pip executable in virtual environment."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"


def install_dependencies(venv_path):
    """Install project dependencies."""
    pip_path = get_venv_pip(venv_path)
    
    print("Upgrading pip...")
    run_command([str(pip_path), "install", "--upgrade", "pip"])
    
    print("Installing build dependencies...")
    run_command([str(pip_path), "install", "--upgrade", "setuptools", "wheel", "build"])
    
    print("Installing project in development mode with all dependencies...")
    run_command([str(pip_path), "install", "-e", ".[dev,audio,examples]"])
    
    print("✓ Dependencies installed")


def setup_pre_commit(venv_path):
    """Set up pre-commit hooks."""
    pip_path = get_venv_pip(venv_path)
    
    print("Installing pre-commit...")
    run_command([str(pip_path), "install", "pre-commit"])
    
    # Check if .pre-commit-config.yaml exists
    pre_commit_config = Path(".pre-commit-config.yaml")
    if not pre_commit_config.exists():
        print("Creating .pre-commit-config.yaml...")
        create_pre_commit_config()
    
    print("Installing pre-commit hooks...")
    venv_python = get_venv_python(venv_path)
    run_command([str(venv_python), "-m", "pre_commit", "install"])
    
    print("✓ Pre-commit hooks installed")


def create_pre_commit_config():
    """Create pre-commit configuration file."""
    config_content = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML]
"""
    
    with open(".pre-commit-config.yaml", "w") as f:
        f.write(config_content)


def create_vscode_settings():
    """Create VS Code settings for the project."""
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    # Settings
    settings_file = vscode_dir / "settings.json"
    if not settings_file.exists():
        settings_content = """{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": [
        "--profile",
        "black"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    }
}"""
        with open(settings_file, "w") as f:
            f.write(settings_content)
        print("✓ VS Code settings created")
    
    # Extensions recommendations
    extensions_file = vscode_dir / "extensions.json"
    if not extensions_file.exists():
        extensions_content = """{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.flake8",
        "ms-python.mypy-type-checker",
        "ms-vscode.test-adapter-converter",
        "littlefoxteam.vscode-python-test-adapter"
    ]
}"""
        with open(extensions_file, "w") as f:
            f.write(extensions_content)
        print("✓ VS Code extensions recommendations created")


def create_example_configs():
    """Create example configuration files."""
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Example VAD configuration
    example_config = configs_dir / "vad_config_example.yaml"
    if not example_config.exists():
        config_content = """# Example VAD Configuration
sample_rate: 16000
model_version: "v5"
vad_start_probability: 0.7
vad_end_probability: 0.7
voice_start_ratio: 0.8
voice_end_ratio: 0.95
voice_start_frame_count: 10
voice_end_frame_count: 57
enable_denoising: true
auto_convert_sample_rate: true
buffer_size: 512
output_wav_sample_rate: 16000
output_wav_bit_depth: 16
"""
        with open(example_config, "w") as f:
            f.write(config_content)
        print("✓ Example VAD configuration created")


def run_tests(venv_path):
    """Run tests to verify installation."""
    venv_python = get_venv_python(venv_path)
    
    print("Running tests to verify installation...")
    try:
        # Run a simple import test
        run_command([
            str(venv_python), "-c",
            "import real_time_vad; print('✓ Package import successful')"
        ])
        
        # Run unit tests (if pytest is available)
        print("Running unit tests...")
        result = run_command([
            str(venv_python), "-m", "pytest",
            "tests/", "-v", "--tb=short", "-m", "not slow"
        ], check=False)
        
        if result.returncode == 0:
            print("✓ Tests passed")
        else:
            print("⚠ Some tests failed (this might be expected if models are not available)")
    
    except Exception as e:
        print(f"⚠ Test execution failed: {e}")


def print_next_steps(venv_path):
    """Print next steps for the user."""
    venv_python = get_venv_python(venv_path)
    
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print(f"1. Activate the virtual environment:")
    
    if platform.system() == "Windows":
        print(f"   .\\venv\\Scripts\\activate")
    else:
        print(f"   source venv/bin/activate")
    
    print(f"\n2. Test the installation:")
    print(f"   python examples/basic_usage.py")
    
    print(f"\n3. Run tests:")
    print(f"   pytest tests/")
    
    print(f"\n4. View documentation:")
    print(f"   Open README.md in your editor")
    
    print(f"\n5. Start developing:")
    print(f"   - Edit files in src/real_time_vad/")
    print(f"   - Add tests in tests/")
    print(f"   - Use pre-commit hooks for code quality")
    
    print(f"\nPython executable: {venv_python}")
    print("Happy coding! 🚀")


def main():
    """Main setup function."""
    print("Real-Time VAD Library - Development Environment Setup")
    print("="*55)
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    try:
        # Check Python version
        check_python_version()
        
        # Create virtual environment
        venv_path = create_virtual_environment()
        
        # Install dependencies
        install_dependencies(venv_path)
        
        # Setup development tools
        setup_pre_commit(venv_path)
        create_vscode_settings()
        create_example_configs()
        
        # Verify installation
        run_tests(venv_path)
        
        # Print next steps
        print_next_steps(venv_path)
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
