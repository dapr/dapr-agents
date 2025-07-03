#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Dapr Agents Local Development Setup${NC}"
echo -e "${BLUE}=====================================${NC}"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "dapr_agents" ]; then
    echo -e "${RED}❌ Please run this script from the dapr-agents directory${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check directory structure
MISSING_DIRS=()
if [ ! -d "../dapr" ]; then
    MISSING_DIRS+=("../dapr")
fi
if [ ! -d "../python-sdk" ]; then
    MISSING_DIRS+=("../python-sdk")
fi

if [ ${#MISSING_DIRS[@]} -gt 0 ]; then
    echo -e "${RED}❌ Missing required directories:${NC}"
    for dir in "${MISSING_DIRS[@]}"; do
        echo -e "   $dir"
    done
    echo
    echo -e "${YELLOW}💡 Please clone the required repositories:${NC}"
    echo "   git clone https://github.com/dapr/dapr.git ../dapr"
    echo "   git clone https://github.com/dapr/python-sdk.git ../python-sdk"
    exit 1
fi

echo -e "${GREEN}✅ Directory structure looks good${NC}"

# Check if Go is available
if ! command -v go &> /dev/null; then
    echo -e "${RED}❌ Go is not installed. Please install Go 1.21+ first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Go is available: $(go version)${NC}"

# Check if Make is available
if ! command -v make &> /dev/null; then
    echo -e "${RED}❌ Make is not installed. Please install Make first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Make is available${NC}"

echo
echo -e "${YELLOW}🔨 Building Dapr runtime...${NC}"
cd ../dapr
if make build TAGS=allcomponents; then
    echo -e "${GREEN}✅ Dapr runtime built successfully${NC}"
else
    echo -e "${RED}❌ Failed to build Dapr runtime${NC}"
    exit 1
fi

cd ../dapr-agents

echo
echo -e "${YELLOW}📦 Installing local Python SDK...${NC}"
# Remove the regular dapr package if it exists to avoid conflicts
pip uninstall dapr -y > /dev/null 2>&1
if pip install -e ../python-sdk; then
    echo -e "${GREEN}✅ Local Python SDK installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install local Python SDK${NC}"
    exit 1
fi

echo
echo -e "${YELLOW}🧪 Verifying installation...${NC}"

# Check Python SDK version
SDK_VERSION=$(python -c "import dapr; print(dapr.__version__)" 2>/dev/null)
if [[ "$SDK_VERSION" == *"dev"* ]]; then
    echo -e "${GREEN}✅ Local Python SDK version: $SDK_VERSION${NC}"
else
    echo -e "${RED}❌ Expected development version, got: $SDK_VERSION${NC}"
    exit 1
fi

echo
echo -e "${GREEN}🎉 Local development setup completed successfully!${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Start local Dapr: ${YELLOW}./start_dapr.sh --dev${NC}"
echo -e "2. Test streaming: ${YELLOW}python test_streaming_with_dapr.py${NC}"
echo -e "3. Read the docs: ${YELLOW}docs/local-development.md${NC}"
echo
echo -e "${BLUE}💡 Pro tip: Use ${YELLOW}./start_dapr.sh --help${NC} to see all available options" 