#!/usr/bin/env python3

"""
Verification script for local development setup.
Run this to check if your local development environment is properly configured.
"""

import sys
import requests
from datetime import datetime


def main():
    print("🧪 Local Development Verification")
    print("=" * 40)
    print(f'⏰ Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    success = True

    # Check SDK version
    try:
        import dapr

        try:
            version = dapr.__version__
        except AttributeError:
            # Try alternative method for version detection
            import pkg_resources

            try:
                version = pkg_resources.get_distribution("dapr-dev").version
            except Exception:
                version = "unknown"

        print(f"📦 Dapr SDK Version: {version}")
        if "dev" in version:
            print("   ✅ Development version detected")
        elif version == "unknown":
            print("   ⚠️  Could not determine version")
        else:
            print("   ⚠️  Not using development version")
            success = False
    except ImportError:
        print("❌ Dapr SDK: Not installed")
        success = False

    # Check streaming methods
    try:
        from dapr.clients import DaprClient

        client = DaprClient()
        stream_methods = [m for m in dir(client) if "stream" in m.lower()]
        print(f"🌊 Streaming Methods: {len(stream_methods)} found")

        if len(stream_methods) >= 2:
            print("   ✅ Streaming methods available:")
            for method in stream_methods:
                print(f"      - {method}")
        else:
            print("   ❌ Streaming methods not found")
            success = False
    except Exception as e:
        print(f"❌ Error checking streaming methods: {e}")
        success = False

    # Check if Dapr is running
    try:
        response = requests.get("http://localhost:3500/v1.0/healthz", timeout=2)
        if response.status_code == 204:
            print("✅ Dapr Sidecar: Running and healthy")
        else:
            print(f"⚠️  Dapr Sidecar: Unexpected response ({response.status_code})")
    except requests.exceptions.ConnectionError:
        print("❌ Dapr Sidecar: Not running")
        print("   💡 Start with: ./start_dapr.sh --dev")
        success = False
    except Exception as e:
        print(f"❌ Error checking Dapr: {e}")
        success = False

    # Check component metadata if Dapr is running
    try:
        response = requests.get("http://localhost:3500/v1.0/metadata", timeout=2)
        if response.status_code == 200:
            metadata = response.json()
            components = metadata.get("components", [])
            echo_component = next(
                (c for c in components if c.get("name") == "echo"), None
            )
            if echo_component:
                print("✅ Echo Component: Loaded and ready")
            else:
                print("⚠️  Echo Component: Not found")
        else:
            print("⚠️  Could not retrieve component metadata")
    except Exception as e:
        print(f"An error occurred: {e}")

    print()
    if success:
        print("🎉 Local development setup is working perfectly!")
        print()
        print("🚀 Ready to test streaming:")
        print("   python test_streaming_with_dapr.py")
    else:
        print("❌ Some issues found with local development setup")
        print()
        print("💡 Troubleshooting:")
        print("   1. Run setup: ./setup-local-dev.sh")
        print("   2. Start Dapr: ./start_dapr.sh --dev")
        print("   3. Check docs: docs/local-development.md")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
