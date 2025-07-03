"""Dapr runtime lifecycle management for testing."""

import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, List
import logging
import requests
import psutil

# Handle imports for both module and direct execution
try:
    from .scenario_manager import DevelopmentScenario
except ImportError:
    # Direct execution - add parent directory to path
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from scenario_manager import DevelopmentScenario

logger = logging.getLogger(__name__)


class DaprManager:
    """Manages Dapr runtime lifecycle for testing scenarios."""

    def __init__(self, use_cli: bool = False):
        self.workspace_root = Path(__file__).parent.parent.parent
        self.dapr_process = None
        self.app_id = None
        self.http_port = 3500
        self.grpc_port = 50001
        self.metrics_port = 9090
        self.components_dir = None
        self.use_cli = use_cli  # Whether to use Dapr CLI vs local binary

    def start_dapr(
        self,
        components_dir: str,
        app_id: str = "test-app",
        scenario: Optional[DevelopmentScenario] = None,
    ) -> bool:
        """Start Dapr runtime with specified components."""
        self.app_id = app_id
        self.components_dir = components_dir

        # Check if Dapr is already running
        if self._is_dapr_running():
            logger.info("Dapr is already running")
            return True

        # Build Dapr command
        cmd = self._build_dapr_command(components_dir, app_id, scenario)

        try:
            logger.info(f"Starting Dapr with command: {' '.join(cmd)}")

            # Start Dapr process
            self.dapr_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Wait for Dapr to be ready
            if self.wait_for_ready():
                logger.info(f"Dapr started successfully with app-id: {app_id}")
                return True
            else:
                logger.error("Dapr failed to start or become ready")
                self.stop_dapr()
                return False

        except Exception as e:
            logger.error(f"Failed to start Dapr: {e}")
            return False

    def _build_dapr_command(
        self,
        components_dir: str,
        app_id: str,
        scenario: Optional[DevelopmentScenario] = None,
    ) -> List[str]:
        """Build Dapr command based on scenario and CLI preference."""
        # Determine which Dapr to use
        dapr_binary = "dapr"  # Default to CLI

        if not self.use_cli and scenario == DevelopmentScenario.LOCAL_FULL:
            # For local full development, try to use local binary
            local_dapr = self._find_local_dapr_binary()
            if local_dapr:
                dapr_binary = str(local_dapr)
                logger.info(f"Using local Dapr binary: {local_dapr}")
            else:
                logger.info("Local Dapr binary not found, falling back to CLI")

        cmd = [
            dapr_binary,
            "run",
            "--app-id",
            app_id,
            "--dapr-http-port",
            str(self.http_port),
            "--dapr-grpc-port",
            str(self.grpc_port),
            "--metrics-port",
            str(self.metrics_port),
            "--resources-path",
            components_dir,
            "--log-level",
            "info",
        ]

        # Add development-friendly settings
        cmd.extend(["--enable-profiling", "--profile-port", "7777"])

        return cmd

    def _find_local_dapr_binary(self) -> Optional[Path]:
        """Find local Dapr binary for development."""
        possible_paths = [
            self.workspace_root.parent
            / "dapr"
            / "dist"
            / "darwin_amd64"
            / "release"
            / "dapr",
            self.workspace_root.parent / "dapr" / "bin" / "dapr",
            self.workspace_root.parent / "dapr" / "dapr",
        ]

        for path in possible_paths:
            if path.exists() and path.is_file():
                return path

        return None

    def stop_dapr(self) -> bool:
        """Stop Dapr runtime."""
        if self.dapr_process:
            try:
                # Try graceful shutdown first
                self.dapr_process.terminate()

                # Wait for process to terminate
                try:
                    self.dapr_process.wait(timeout=10)
                    logger.info("Dapr stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.dapr_process.kill()
                    self.dapr_process.wait()
                    logger.warning("Dapr force killed")

                self.dapr_process = None
                return True

            except Exception as e:
                logger.error(f"Error stopping Dapr: {e}")
                return False

        # Also try to stop any other Dapr processes
        return self._cleanup_dapr_processes()

    def _cleanup_dapr_processes(self) -> bool:
        """Clean up any remaining Dapr processes."""
        try:
            # Find Dapr processes
            dapr_processes = []
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if proc.info["name"] == "dapr" or any(
                        "dapr" in str(arg) for arg in proc.info["cmdline"] or []
                    ):
                        dapr_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Terminate processes
            for proc in dapr_processes:
                try:
                    proc.terminate()
                    logger.info(f"Terminated Dapr process: {proc.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Wait and force kill if needed
            time.sleep(2)
            for proc in dapr_processes:
                try:
                    if proc.is_running():
                        proc.kill()
                        logger.warning(f"Force killed Dapr process: {proc.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return True

        except Exception as e:
            logger.error(f"Error cleaning up Dapr processes: {e}")
            return False

    def wait_for_ready(self, timeout: int = 5) -> bool:
        """Wait for Dapr to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._check_dapr_health():
                return True

            # Check if process is still running
            if self.dapr_process and self.dapr_process.poll() is not None:
                logger.error("Dapr process exited unexpectedly")
                return False

            time.sleep(0.5)  # Check more frequently for faster feedback

        logger.error(f"Dapr not ready after {timeout} seconds")
        return False

    def _check_dapr_health(self) -> bool:
        """Check if Dapr is healthy."""
        try:
            # Check HTTP endpoint - Accept any 2xx status code (200-299)
            response = requests.get(
                f"http://localhost:{self.http_port}/v1.0/healthz", timeout=5
            )
            return 200 <= response.status_code < 300
        except requests.RequestException:
            return False

    def _is_dapr_running(self) -> bool:
        """Check if Dapr is already running."""
        return self._check_dapr_health()

    def get_endpoints(self) -> Dict[str, str]:
        """Get Dapr endpoints."""
        return {
            "http": f"http://localhost:{self.http_port}",
            "grpc": f"localhost:{self.grpc_port}",
            "metrics": f"http://localhost:{self.metrics_port}",
        }

    def get_metadata(self) -> Optional[Dict]:
        """Get Dapr metadata."""
        try:
            response = requests.get(
                f"http://localhost:{self.http_port}/v1.0/metadata", timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as e:
            logger.warning(f"Could not get Dapr metadata: {e}")

        return None

    def list_components(self) -> List[str]:
        """List loaded components."""
        metadata = self.get_metadata()
        if metadata and "components" in metadata:
            return [comp["name"] for comp in metadata["components"]]
        return []

    def restart_dapr(self) -> bool:
        """Restart Dapr runtime."""
        logger.info("Restarting Dapr...")

        # Stop current instance
        if not self.stop_dapr():
            logger.error("Failed to stop Dapr")
            return False

        # Wait a bit for cleanup
        time.sleep(2)

        # Start again with same configuration
        if self.components_dir and self.app_id:
            return self.start_dapr(self.components_dir, self.app_id)
        else:
            logger.error("Cannot restart - missing configuration")
            return False

    def get_logs(self) -> Optional[str]:
        """Get Dapr logs."""
        if self.dapr_process:
            try:
                # Get stdout and stderr
                stdout, stderr = self.dapr_process.communicate(timeout=1)
                logs = ""
                if stdout:
                    logs += "STDOUT:\n" + stdout + "\n"
                if stderr:
                    logs += "STDERR:\n" + stderr + "\n"
                return logs
            except subprocess.TimeoutExpired:
                # Process is still running, can't get logs this way
                pass

        return None

    def check_ports_available(self) -> Dict[str, bool]:
        """Check if required ports are available."""
        ports = {
            "http": self.http_port,
            "grpc": self.grpc_port,
            "metrics": self.metrics_port,
        }

        availability = {}
        for name, port in ports.items():
            availability[name] = self._is_port_available(port)

        return availability

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result != 0  # Port is available if connection fails
        except Exception:
            return False

    def get_status_summary(self) -> str:
        """Get formatted status summary."""
        summary = "Dapr Manager Status:\n"
        summary += f"  App ID: {self.app_id or 'Not set'}\n"
        summary += f"  Components Dir: {self.components_dir or 'Not set'}\n"

        # Process status
        if self.dapr_process:
            if self.dapr_process.poll() is None:
                summary += f"  Process: Running (PID: {self.dapr_process.pid})\n"
            else:
                summary += (
                    f"  Process: Stopped (Exit code: {self.dapr_process.poll()})\n"
                )
        else:
            summary += "  Process: Not started\n"

        # Health status
        is_healthy = self._check_dapr_health()
        summary += f"  Health: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}\n"

        # Endpoints
        endpoints = self.get_endpoints()
        summary += f"  HTTP Endpoint: {endpoints['http']}\n"
        summary += f"  gRPC Endpoint: {endpoints['grpc']}\n"

        # Components
        if is_healthy:
            components = self.list_components()
            summary += f"  Components: {len(components)} loaded\n"
            for comp in components[:3]:  # Show first 3
                summary += f"    - {comp}\n"
            if len(components) > 3:
                summary += f"    ... and {len(components) - 3} more\n"

        return summary

    # CLI Support Methods

    def _is_dapr_cli_installed(self) -> bool:
        """Check if Dapr CLI is installed."""
        try:
            result = subprocess.run(
                ["dapr", "--version"], capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _get_dapr_cli_version(self) -> Optional[str]:
        """Get installed Dapr CLI version."""
        try:
            result = subprocess.run(
                ["dapr", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # Parse version from output like "CLI version: 1.15.1"
                for line in result.stdout.split("\n"):
                    if "CLI version:" in line:
                        return line.split(":")[1].strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _install_dapr_cli(self, interactive: bool = True) -> bool:
        """Install Dapr CLI with user confirmation."""
        if not interactive:
            logger.info("Non-interactive mode: skipping Dapr CLI installation")
            return False

        print("\n🚀 Dapr CLI is not installed!")
        print("The Dapr CLI is required for testing with production-like setup.")
        print(
            "It includes Redis (required) and Zipkin (optional tracing) for state management and observability."
        )
        print("\nInstallation methods:")
        print("  1. Automatic (recommended)")
        print("  2. Manual instructions")
        print("  3. Skip installation")

        while True:
            choice = input("\nChoose option (1/2/3): ").strip()

            if choice == "1":
                return self._auto_install_dapr_cli()
            elif choice == "2":
                self._show_manual_installation_instructions()
                return False
            elif choice == "3":
                print("Skipping Dapr CLI installation.")
                return False
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    def _auto_install_dapr_cli(self) -> bool:
        """Automatically install Dapr CLI."""
        print("\n📦 Installing Dapr CLI...")

        try:
            # Detect OS
            import platform

            system = platform.system().lower()

            if system == "darwin":  # macOS
                print("🍎 Detected macOS - installing via Homebrew...")
                # Check if Homebrew is installed
                brew_check = subprocess.run(
                    ["which", "brew"], capture_output=True, text=True
                )
                if brew_check.returncode != 0:
                    print("❌ Homebrew not found. Please install Homebrew first:")
                    print(
                        '   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                    )
                    return False

                # Install Dapr CLI via Homebrew
                result = subprocess.run(
                    ["brew", "install", "dapr/tap/dapr-cli"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    print("✅ Dapr CLI installed successfully!")
                    return True
                else:
                    print(f"❌ Installation failed: {result.stderr}")
                    return False

            elif system == "linux":
                print("🐧 Detected Linux - installing via curl...")
                # Use the official install script
                install_cmd = [
                    "bash",
                    "-c",
                    "wget -q https://raw.githubusercontent.com/dapr/cli/master/install/install.sh -O - | /bin/bash",
                ]
                result = subprocess.run(
                    install_cmd, capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    print("✅ Dapr CLI installed successfully!")
                    print("💡 You may need to add ~/.dapr/bin to your PATH")
                    return True
                else:
                    print(f"❌ Installation failed: {result.stderr}")
                    return False

            else:
                print(f"❌ Unsupported OS: {system}")
                self._show_manual_installation_instructions()
                return False

        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False

    def _show_manual_installation_instructions(self):
        """Show manual installation instructions."""
        print("\n📋 Manual Installation Instructions:")
        print("\n🍎 macOS:")
        print("   brew install dapr/tap/dapr-cli")
        print("\n🐧 Linux:")
        print(
            "   wget -q https://raw.githubusercontent.com/dapr/cli/master/install/install.sh -O - | /bin/bash"
        )
        print("\n🪟 Windows:")
        print(
            '   powershell -Command "iwr -useb https://raw.githubusercontent.com/dapr/cli/master/install/install.ps1 | iex"'
        )
        print("\n📖 More info: https://docs.dapr.io/getting-started/install-dapr-cli/")

    def _is_dapr_initialized(self) -> bool:
        """Check if Dapr is initialized (core containers running)."""
        try:
            # Check if Docker containers are running
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=dapr", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Only require Redis (core requirement) - Zipkin is optional for observability
                container_names = result.stdout.lower()
                return "dapr_redis" in container_names
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Docker not available or timeout
            return False

    def _initialize_dapr(self, interactive: bool = True) -> bool:
        """Initialize Dapr with Redis and Zipkin."""
        if self._is_dapr_initialized():
            logger.info("Dapr is already initialized")
            return True

        if interactive:
            print("\n🔧 Dapr needs to be initialized!")
            print(
                "This will download and start Redis (required) and Zipkin (optional for tracing)."
            )
            confirm = input("Initialize Dapr? (y/N): ").strip().lower()
            if confirm not in ["y", "yes"]:
                print("Skipping Dapr initialization.")
                return False

        print("\n⚙️ Initializing Dapr...")
        try:
            result = subprocess.run(
                ["dapr", "init"], capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print("✅ Dapr initialized successfully!")
                print(
                    "🐳 Redis (required) and Zipkin (optional tracing) containers are now running"
                )
                return True
            else:
                print(f"❌ Initialization failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Initialization timed out")
            return False
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return False

    def setup_dapr_cli(self, interactive: bool = True) -> bool:
        """Complete setup of Dapr CLI including installation and initialization."""
        logger.info("Setting up Dapr CLI...")

        # Check if CLI is installed
        if not self._is_dapr_cli_installed():
            logger.info("Dapr CLI not found - attempting installation")
            if not self._install_dapr_cli(interactive):
                return False
        else:
            version = self._get_dapr_cli_version()
            logger.info(f"Dapr CLI already installed: {version}")

        # Check if Dapr is initialized
        if not self._is_dapr_initialized():
            logger.info("Dapr not initialized - attempting initialization")
            if not self._initialize_dapr(interactive):
                return False
        else:
            logger.info("Dapr already initialized")

        return True


if __name__ == "__main__":
    # CLI interface for Dapr management
    import argparse

    parser = argparse.ArgumentParser(description="Manage Dapr runtime for testing")
    parser.add_argument("--start", help="Start Dapr with components directory")
    parser.add_argument("--stop", action="store_true", help="Stop Dapr")
    parser.add_argument("--restart", action="store_true", help="Restart Dapr")
    parser.add_argument("--status", action="store_true", help="Show Dapr status")
    parser.add_argument("--app-id", default="test-app", help="App ID for Dapr")
    parser.add_argument(
        "--check-ports", action="store_true", help="Check port availability"
    )
    parser.add_argument(
        "--use-cli", action="store_true", help="Use Dapr CLI instead of local binary"
    )
    parser.add_argument(
        "--setup-cli",
        action="store_true",
        help="Setup Dapr CLI (install and initialize)",
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Non-interactive mode"
    )
    args = parser.parse_args()

    manager = DaprManager(use_cli=args.use_cli)

    if args.setup_cli:
        interactive = not args.non_interactive
        success = manager.setup_dapr_cli(interactive=interactive)
        print(f"Dapr CLI setup: {'Success' if success else 'Failed'}")

    elif args.start:
        success = manager.start_dapr(args.start, args.app_id)
        print(f"Dapr start: {'Success' if success else 'Failed'}")

    elif args.stop:
        success = manager.stop_dapr()
        print(f"Dapr stop: {'Success' if success else 'Failed'}")

    elif args.restart:
        success = manager.restart_dapr()
        print(f"Dapr restart: {'Success' if success else 'Failed'}")

    elif args.check_ports:
        ports = manager.check_ports_available()
        print("Port Availability:")
        for name, available in ports.items():
            status = "Available" if available else "In use"
            print(f"  {name}: {status}")

    elif args.status:
        print(manager.get_status_summary())

        # Also show CLI info with detailed container status
        if manager._is_dapr_cli_installed():
            version = manager._get_dapr_cli_version()
            print(f"\nDapr CLI: ✓ Installed ({version})")

            # Check individual containers
            try:
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=dapr", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    container_names = result.stdout.lower()
                    redis_running = "dapr_redis" in container_names
                    zipkin_running = "dapr_zipkin" in container_names
                    placement_running = "dapr_placement" in container_names

                    print(
                        f"Dapr Init: {'✓' if redis_running else '✗'} Redis (required): {'Running' if redis_running else 'Not running'}"
                    )
                    print(
                        f"           {'✓' if zipkin_running else '○'} Zipkin (optional): {'Running' if zipkin_running else 'Not running'}"
                    )
                    print(
                        f"           {'✓' if placement_running else '○'} Placement (optional): {'Running' if placement_running else 'Not running'}"
                    )

                    if redis_running:
                        print("           ✓ Dapr is ready for basic functionality")
                    else:
                        print("           ✗ Dapr needs Redis for state management")
                else:
                    print("Dapr Init: ✗ Cannot check containers (Docker not available)")
            except Exception:
                print("Dapr Init: ✗ Cannot check containers")
        else:
            print("\nDapr CLI: ✗ Not installed")

    else:
        print("Use --help for available options")
