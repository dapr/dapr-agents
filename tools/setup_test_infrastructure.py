#!/usr/bin/env python3
"""
Dapr Agents Test Infrastructure Setup Script

This script automates the setup of the complete test infrastructure for Dapr Agents,
including Dapr initialization, environment validation, and component verification.
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import List


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_step(message: str, step_num: int = None):
    """Print a setup step with formatting"""
    if step_num:
        print(f"\n{Colors.BLUE}{Colors.BOLD}Step {step_num}: {message}{Colors.END}")
    else:
        print(f"{Colors.BLUE}{message}{Colors.END}")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def run_command(
    cmd: List[str], capture_output: bool = True, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            cmd, capture_output=capture_output, text=True, check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        if capture_output:
            print_error(f"Command failed: {' '.join(cmd)}")
            print_error(f"Error: {e.stderr}")
        raise


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH"""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_docker_running() -> bool:
    """Check if Docker is running"""
    try:
        subprocess.run(["docker", "ps"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_dapr_cli() -> bool:
    """Install Dapr CLI if not present"""
    if check_command_exists("dapr"):
        print_success("Dapr CLI already installed")
        return True

    print_step("Installing Dapr CLI...")

    # Detect OS
    import platform

    system = platform.system().lower()

    try:
        if system == "darwin":  # macOS
            cmd = [
                "curl",
                "-fsSL",
                "https://raw.githubusercontent.com/dapr/cli/master/install/install.sh",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            subprocess.run(["bash"], input=result.stdout, text=True, check=True)
        elif system == "linux":
            cmd = [
                "wget",
                "-q",
                "https://raw.githubusercontent.com/dapr/cli/master/install/install.sh",
                "-O",
                "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            subprocess.run(["bash"], input=result.stdout, text=True, check=True)
        else:
            print_error(f"Unsupported OS: {system}")
            print_error(
                "Please install Dapr CLI manually: https://docs.dapr.io/getting-started/install-dapr-cli/"
            )
            return False

        # Verify installation
        if check_command_exists("dapr"):
            print_success("Dapr CLI installed successfully")
            return True
        else:
            print_error("Dapr CLI installation failed")
            return False

    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install Dapr CLI: {e}")
        return False


def initialize_dapr() -> bool:
    """Initialize Dapr infrastructure"""
    print_step("Initializing Dapr infrastructure...")

    try:
        # Check if already initialized
        result = run_command(["dapr", "status"], check=False)
        if result.returncode == 0 and "dapr_placement" in result.stdout:
            print_success("Dapr already initialized")
            return True

        # Initialize Dapr
        print("Running 'dapr init'...")
        run_command(["dapr", "init"], capture_output=False)

        # Wait for services to start
        print("Waiting for Dapr services to start...")
        time.sleep(10)

        # Verify initialization
        result = run_command(["dapr", "status"])
        if "dapr_placement" in result.stdout:
            print_success("Dapr initialized successfully")
            return True
        else:
            print_error("Dapr initialization verification failed")
            return False

    except subprocess.CalledProcessError as e:
        print_error(f"Failed to initialize Dapr: {e}")
        return False


def verify_docker_containers() -> bool:
    """Verify required Docker containers are running"""
    print_step("Verifying Docker containers...")

    required_containers = [
        "dapr_placement",
        "dapr_redis",
        "dapr_zipkin",
        "dapr_scheduler",
    ]

    try:
        result = run_command(["docker", "ps", "--format", "table {{.Names}}"])
        running_containers = result.stdout

        missing_containers = []
        for container in required_containers:
            if container in running_containers:
                print_success(f"{container} is running")
            else:
                missing_containers.append(container)
                print_warning(f"{container} is not running")

        if missing_containers:
            print_error(f"Missing containers: {', '.join(missing_containers)}")
            print_error("Try running 'dapr uninstall' followed by 'dapr init'")
            return False

        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Failed to check Docker containers: {e}")
        return False


def check_environment_file() -> bool:
    """Check for .env file and create template if missing"""
    print_step("Checking environment configuration...")

    env_file = Path(".env")
    if env_file.exists():
        print_success(".env file found")
        return True

    print_warning(".env file not found, creating template...")

    env_template = """# Dapr Agents Environment Configuration
#
# Required for real provider testing (optional for echo testing)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_AI_API_KEY=your_gemini_api_key_here

# Optional for enhanced testing
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
NVIDIA_API_KEY=your_nvidia_key_here

# Development settings
DAPR_HTTP_PORT=3500
DAPR_GRPC_PORT=50001
LOG_LEVEL=info
"""

    try:
        env_file.write_text(env_template)
        print_success("Created .env template file")
        print_warning("Please edit .env with your API keys for full testing")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False


def verify_components() -> bool:
    """Verify component files exist"""
    print_step("Verifying component files...")

    components_dir = Path("components")
    if not components_dir.exists():
        print_error("Components directory not found")
        return False

    required_components = [
        "echo-conversation.yaml",
        "echo-tool-calling.yaml",
        "anthropic-conversation.yaml",
        "gemini-conversation.yaml",
        "openai-conversation.yaml",
        "conversationstore.yaml",
        "registrystatestore.yaml",
        "redis-pubsub.yaml",
        "redis-statestore.yaml",
    ]

    missing_components = []
    for component in required_components:
        component_file = components_dir / component
        if component_file.exists():
            print_success(f"Found {component}")
        else:
            missing_components.append(component)
            print_warning(f"Missing {component}")

    if missing_components:
        print_error(f"Missing components: {', '.join(missing_components)}")
        return False

    print_success("All component files found")
    return True


def check_python_dependencies() -> bool:
    """Check if required Python dependencies are installed"""
    print_step("Checking Python dependencies...")

    required_packages = ["pytest", "dapr", "requests", "pydantic"]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            missing_packages.append(package)
            print_warning(f"{package} is not installed")

    if missing_packages:
        print_error(f"Missing packages: {', '.join(missing_packages)}")
        print_error("Run: pip install -r requirements.txt")
        print_error("Run: pip install -r tests/requirements-test.txt")
        return False

    return True


def test_basic_functionality() -> bool:
    """Test basic functionality with echo provider"""
    print_step("Testing basic functionality...")

    try:
        # Start Dapr sidecar in background
        print("Starting Dapr sidecar for testing...")
        dapr_process = subprocess.Popen(
            [
                "python",
                "tools/run_dapr_dev.py",
                "--app-id",
                "setup-test",
                "--components",
                "./components",
                "--log-level",
                "info",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for startup
        time.sleep(8)

        # Check if process is still running
        if dapr_process.poll() is not None:
            stdout, stderr = dapr_process.communicate()
            print_error("Dapr sidecar failed to start")
            print_error(f"Error: {stderr.decode()}")
            return False

        # Run a simple test
        print("Running basic echo test...")
        test_result = run_command(
            [
                "python",
                "-m",
                "pytest",
                "tests/integration/test_basic_integration.py::test_echo_conversation",
                "-v",
                "--tb=short",
            ],
            capture_output=True,
            check=False,
        )

        # Cleanup
        dapr_process.terminate()
        dapr_process.wait(timeout=10)

        if test_result.returncode == 0:
            print_success("Basic functionality test passed")
            return True
        else:
            print_error("Basic functionality test failed")
            print_error(f"Test output: {test_result.stdout}")
            print_error(f"Test errors: {test_result.stderr}")
            return False

    except Exception as e:
        # Cleanup on error
        if "dapr_process" in locals():
            dapr_process.terminate()
        print_error(f"Failed to test basic functionality: {e}")
        return False


def main():
    """Main setup function"""
    print(f"{Colors.BOLD}🧪 Dapr Agents Test Infrastructure Setup{Colors.END}")
    print("=" * 50)

    # Check prerequisites
    print_step("Checking prerequisites...", 1)

    if not check_docker_running():
        print_error("Docker is not running. Please start Docker and try again.")
        sys.exit(1)
    print_success("Docker is running")

    # Install Dapr CLI
    if not install_dapr_cli():
        print_error("Failed to install Dapr CLI")
        sys.exit(1)

    # Initialize Dapr
    if not initialize_dapr():
        print_error("Failed to initialize Dapr")
        sys.exit(1)

    # Verify Docker containers
    if not verify_docker_containers():
        print_error("Docker container verification failed")
        sys.exit(1)

    # Check environment
    print_step("Setting up environment...", 2)
    check_environment_file()  # This creates template if missing

    # Verify components
    if not verify_components():
        print_error("Component verification failed")
        sys.exit(1)

    # Check Python dependencies
    if not check_python_dependencies():
        print_error("Python dependency check failed")
        print_error("Please install missing dependencies and run setup again")
        sys.exit(1)

    # Test basic functionality
    print_step("Testing infrastructure...", 3)
    if not test_basic_functionality():
        print_error("Basic functionality test failed")
        print_warning("Infrastructure setup completed but testing failed")
        print_warning("Check the troubleshooting section in tests/README.md")
        sys.exit(1)

    # Success
    print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Setup completed successfully!{Colors.END}")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys (optional)")
    print("2. Run tests: cd tests && python -m pytest integration/ -v")
    print("3. See tests/README.md for detailed usage instructions")

    print(f"\n{Colors.BLUE}Quick test commands:{Colors.END}")
    print("  # Fast smoke test")
    print(
        "  cd tests && python -m pytest integration/test_basic_integration.py::test_echo_conversation -v"
    )
    print("\n  # All chat scenarios")
    print("  cd tests && python -m pytest integration/test_chat_scenarios.py -v -s")
    print("\n  # React agent tests")
    print(
        "  cd tests && python -m pytest integration/test_react_agent_scenarios.py -v -s"
    )


if __name__ == "__main__":
    main()
