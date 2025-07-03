#!/usr/bin/env python3

"""
Development helper for running Dapr sidecar with conversation components.

This script helps developers quickly start a Dapr sidecar with conversation components
for testing the Python SDK conversation streaming functionality.

Usage:
    python tools/run_dapr_dev.py [options]

Options:
    --build         Build daprd binary before running
    --port          HTTP port for sidecar (default: 3500)
    --grpc-port     gRPC port for sidecar (default: 50001)
    --app-id        Application ID (default: test-app)
    --log-level     Log level (default: info)
    --components    Path to components directory (default: auto-created)
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

# make sure dapr repo is in the right place
DAPR_REPO = REPO_ROOT.parent / "dapr"
if not DAPR_REPO.exists():
    print(f"❌ Error: Dapr repository not found at {DAPR_REPO}")
    print(
        "Please clone the dapr repository at ../dapr relative to python-sdk and dapr-agents"
    )
    sys.exit(1)


def load_env_file():
    """Load environment variables from .env file at repo root."""
    env_file = REPO_ROOT / ".env"
    env_vars = {}

    if env_file.exists():
        print(f"📄 Loading environment variables from {env_file}")
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")  # Remove quotes
                            env_vars[key] = value
                            # Also set in os.environ for subprocess
                            os.environ[key] = value
            print(f"✅ Loaded {len(env_vars)} environment variables")
        except Exception as e:
            print(f"⚠️  Warning: Could not read .env file: {e}")
    else:
        print(f"⚠️  Warning: No .env file found at {env_file}")

    providers = [
        "GEMINI",
        "OPENAI",
        "ANTHROPIC",
        "DEEPSEEK",
        "MISTRAL",
    ]

    providers_alt_keys = {
        "GEMINI": ["GOOGLE", "GOOGLE_AI", "GEMINI"],
    }

    # show what LLM providers we have keys for (don't show the keys)
    print("🔑 LLM Providers with keys:")
    for provider in providers:
        key = get_provider_key(provider, env_vars, providers_alt_keys)
        if key:
            print(f"   - {provider}")
        else:
            print(f"   - {provider} (not found in .env or environment)")

    return env_vars


def get_provider_key(provider, env_vars, providers_alt_keys):
    """Get the API key for a provider."""
    provider_prefixes = providers_alt_keys.get(provider, [provider])
    value = None
    for prefix in provider_prefixes:
        key = f"{prefix}_API_KEY"
        if key in env_vars:
            value = env_vars[key]
            env_vars[key] = value
            break
        elif key in os.environ:
            value = os.environ[key]
            break
    if value:
        for prefix in provider_prefixes:
            env_vars[f"{prefix}_API_KEY"] = value
        return value
    else:
        return None


def process_component_file(source_file, target_file, env_vars):
    """Process a component file and replace environment variable placeholders."""
    try:
        with open(source_file, "r") as f:
            content = f.read()

        # Replace ${VAR_NAME} patterns with actual values
        def replace_env_var(match):
            var_name = match.group(1)
            if var_name in env_vars:
                return env_vars[var_name]
            elif var_name in os.environ:
                return os.environ[var_name]
            else:
                print(
                    f"⚠️  Warning: Environment variable {var_name} not found for {source_file.name}"
                )
                return match.group(0)  # Return original if not found

        # Replace ${VAR_NAME} patterns
        processed_content = re.sub(r"\$\{([^}]+)\}", replace_env_var, content)

        # Write processed content to target
        with open(target_file, "w") as f:
            f.write(processed_content)

        return True
    except Exception as e:
        print(f"❌ Error processing {source_file}: {e}")
        return False


def prepare_components(components_dir, env_vars):
    """Prepare components by processing .env variables and copying to temp directory."""
    source_components_dir = Path(components_dir)
    temp_dir = tempfile.mkdtemp(prefix="dapr-dev-components-")
    temp_components_dir = Path(temp_dir)

    print(
        f"📁 Processing components from {source_components_dir} to {temp_components_dir}"
    )

    processed_count = 0
    skipped_count = 0

    for component_file in source_components_dir.glob("*.yaml"):
        # Skip disabled files
        if component_file.name.endswith(".disabled"):
            print(f"⏭️  Skipping disabled component: {component_file.name}")
            skipped_count += 1
            continue

        target_file = temp_components_dir / component_file.name

        if process_component_file(component_file, target_file, env_vars):
            print(f"✅ Processed component: {component_file.name}")
            processed_count += 1
        else:
            skipped_count += 1

    print(
        f"📊 Component processing summary: {processed_count} processed, {skipped_count} skipped"
    )

    if processed_count == 0:
        print("❌ No components were successfully processed!")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    return temp_components_dir


def check_dapr_repo():
    """Check if local dapr repository exists."""
    if not DAPR_REPO.exists():
        print(f"❌ Error: Dapr repository not found at {DAPR_REPO}")
        print("Please clone the dapr repository at ../dapr relative to python-sdk")
        sys.exit(1)
    return True


def build_daprd():
    """Build the daprd binary."""
    print("🔨 Building daprd binary...")
    try:
        subprocess.run(
            ["make", "build"], cwd=DAPR_REPO, check=True, capture_output=True, text=True
        )
        print("✅ Build successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def get_daprd_binary():
    """Get the path to the daprd binary."""
    # Check for platform-specific binary
    import platform

    system = platform.system().lower()
    arch = platform.machine().lower()

    if arch == "x86_64":
        arch = "amd64"
    elif arch in ["arm64", "aarch64"]:
        arch = "arm64"

    binary_path = DAPR_REPO / "dist" / f"{system}_{arch}" / "release" / "daprd"

    if binary_path.exists():
        return binary_path

    # Fallback to root directory
    fallback_path = DAPR_REPO / "daprd"
    if fallback_path.exists():
        return fallback_path

    print(f"❌ Error: daprd binary not found at {binary_path} or {fallback_path}")
    print("Try running with --build to build the binary")
    sys.exit(1)


def create_conversation_components(components_dir):
    """Create conversation component configuration."""
    components_dir = Path(components_dir)
    components_dir.mkdir(exist_ok=True)

    echo_component = components_dir / "echo-conversation.yaml"
    echo_component.write_text(
        """apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: echo
spec:
  type: conversation.echo
  version: v1
  metadata:
  - name: key
    value: testkey
"""
    )

    print(f"📝 Created conversation components in {components_dir}")
    return components_dir


def run_daprd(args):
    """Run the daprd sidecar."""
    binary_path = get_daprd_binary()

    # Load environment variables from .env file
    env_vars = load_env_file()
    # Special mapping for Gemini - it expects GEMINI_API_KEY but we have GOOGLE_AI_API_KEY
    if "GOOGLE_AI_API_KEY" in env_vars:
        os.environ["GEMINI_API_KEY"] = env_vars["GOOGLE_AI_API_KEY"]

    # Process components directory if specified
    if args.components:
        components_dir = prepare_components(args.components, env_vars)
        if not components_dir:
            print("❌ Failed to process components directory")
            sys.exit(1)
        temp_dir = str(components_dir.parent)  # For cleanup
    else:
        # Create temporary components directory with default echo component
        temp_dir = tempfile.mkdtemp(prefix="dapr-dev-")
        components_dir = create_conversation_components(temp_dir)

    cmd = [
        str(binary_path),
        "--app-id",
        args.app_id,
        "--dapr-http-port",
        str(args.port),
        "--dapr-grpc-port",
        str(args.grpc_port),
        "--log-level",
        args.log_level,
        "--enable-app-health-check=false",
        "--resources-path",
        str(components_dir),
        "--placement-host-address",
        "localhost:50005",
        "--metrics-port",
        "9091",
    ]

    # check if dapr-config.yaml exists in the components directory, if it is we will add --config to the command
    if (components_dir / "dapr-config.yaml").exists():
        cmd.append("--config")
        cmd.append(str(components_dir / "dapr-config.yaml"))

    print("🚀 Starting Dapr sidecar...")
    print(f"   Binary: {binary_path}")
    print(f"   App ID: {args.app_id}")
    print(f"   HTTP Port: {args.port}")
    print(f"   gRPC Port: {args.grpc_port}")
    print(f"   Components: {components_dir}")
    print(f"   Command: {' '.join(cmd)}")
    print("\n📡 Sidecar output:")
    print("-" * 50)

    try:
        # Run the sidecar with explicit environment inheritance
        # Set environment variables for the process
        env = os.environ.copy()
        env.update(env_vars)
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n🛑 Stopping sidecar...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Sidecar failed: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary directory
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"🧹 Cleaned up temporary directory: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Dapr sidecar for development")
    parser.add_argument(
        "--build", action="store_true", help="Build daprd binary before running"
    )
    parser.add_argument(
        "--port", type=int, default=3500, help="HTTP port (default: 3500)"
    )
    parser.add_argument(
        "--grpc-port", type=int, default=50001, help="gRPC port (default: 50001)"
    )
    parser.add_argument(
        "--app-id", default="test-app", help="Application ID (default: test-app)"
    )
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    parser.add_argument("--components", help="Path to components directory")

    args = parser.parse_args()

    print("🧪 Dapr Development Helper")
    print("=" * 40)

    # Check prerequisites
    check_dapr_repo()

    # Build if requested
    if args.build:
        if not build_daprd():
            sys.exit(1)

    # Run the sidecar
    run_daprd(args)


if __name__ == "__main__":
    main()
