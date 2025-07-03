# Quick Start: Local Development Setup

🚀 **Get up and running with local Dapr development in 5 minutes**

## Prerequisites

```bash
# Required directory structure
parent-directory/
├── dapr/           # git clone https://github.com/dapr/dapr.git
├── dapr-cli/       # git clone https://github.com/dapr/cli.git dapr-cli  
├── python-sdk/     # git clone https://github.com/dapr/python-sdk.git
└── dapr-agents/    # This project
```

## One-Command Setup

```bash
# From dapr-agents directory
./setup-local-dev.sh
```

**Or manually:**

```bash
# 1. Build Dapr runtime
cd ../dapr && make build TAGS=allcomponents

# 2. Install local Python SDK
cd ../dapr-agents
pip uninstall dapr -y && pip install -e ../python-sdk

# 3. Start local Dapr
./start_dapr.sh --dev
```

## Verify Setup

```bash
# Check for streaming methods (should show 2 methods)
python -c "from dapr.clients import DaprClient; print([m for m in dir(DaprClient()) if 'stream' in m])"

# Test streaming conversation
python test_streaming_with_dapr.py
```

## Key Commands

| Command | Purpose |
|---------|---------|
| `./start_dapr.sh --dev` | Start local development Dapr |
| `./start_dapr.sh --release` | Start regular release Dapr |
| `./start_dapr.sh --dev --build` | Build and start local Dapr |

## What You Get

✅ **Streaming Conversation API** - `converse_stream_alpha1()`  
✅ **Latest Dapr Features** - Development version access  
✅ **Full Source Debugging** - Complete visibility  
✅ **Rapid Iteration** - Test changes immediately  

## Need Help?

- 📖 **Full Guide**: [Local Development Setup](local-development.md)
- 🧪 **Test Streaming**: `python test_streaming_with_dapr.py`
- 🔧 **Troubleshooting**: Check the full documentation for common issues

---

**Ready to build with cutting-edge Dapr features? Start with local development! 🚀** 