#!/bin/bash

# Configuration
DEV_DAPR_DIR="../dapr"
DEV_CLI_DIR="../dapr-cli"
DEV_DAPRD_BINARY="$DEV_DAPR_DIR/dist/darwin_arm64/release/daprd"
DEV_CLI_BINARY="$DEV_CLI_DIR/cli"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo -e "${BLUE}🚀 Dapr Sidecar Launcher${NC}"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -d, --dev      Use local development Dapr (from ../dapr)"
    echo "  -r, --release  Use regular release Dapr (default)"
    echo "  -b, --build    Build local Dapr before running (only with --dev)"
    echo "  -h, --help     Show this help message"
    echo
    echo "Examples:"
    echo "  $0                    # Use regular release Dapr"
    echo "  $0 --dev              # Use local development Dapr"
    echo "  $0 --dev --build      # Build and use local development Dapr"
    exit 0
}

build_dev_dapr() {
    echo -e "${YELLOW}🔨 Building local development Dapr...${NC}"
    
    # Build daprd
    if [ -d "$DEV_DAPR_DIR" ]; then
        echo -e "${BLUE}Building daprd...${NC}"
        cd "$DEV_DAPR_DIR"
        if make build; then
            echo -e "${GREEN}✅ daprd built successfully${NC}"
        else
            echo -e "${RED}❌ Failed to build daprd${NC}"
            exit 1
        fi
        cd - > /dev/null
    else
        echo -e "${RED}❌ Dapr directory not found: $DEV_DAPR_DIR${NC}"
        exit 1
    fi
    
    # Build CLI
    if [ -d "$DEV_CLI_DIR" ]; then
        echo -e "${BLUE}Building Dapr CLI...${NC}"
        cd "$DEV_CLI_DIR"
        if make build; then
            echo -e "${GREEN}✅ Dapr CLI built successfully${NC}"
        else
            echo -e "${RED}❌ Failed to build Dapr CLI${NC}"
            exit 1
        fi
        cd - > /dev/null
    else
        echo -e "${RED}❌ Dapr CLI directory not found: $DEV_CLI_DIR${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}🎉 All components built successfully!${NC}"
}

check_dev_binaries() {
    local missing=false
    
    if [ ! -f "$DEV_DAPRD_BINARY" ]; then
        echo -e "${RED}❌ daprd binary not found: $DEV_DAPRD_BINARY${NC}"
        missing=true
    fi
    
    if [ ! -f "$DEV_CLI_BINARY" ]; then
        echo -e "${RED}❌ Dapr CLI binary not found: $DEV_CLI_BINARY${NC}"
        missing=true
    fi
    
    if [ "$missing" = true ]; then
        echo -e "${YELLOW}💡 Try running with --build flag to build the binaries${NC}"
        exit 1
    fi
}

run_with_dev_dapr() {
    echo -e "${BLUE}🚀 Starting Dapr Sidecar with Local Development Binaries${NC}"
    echo -e "${BLUE}Using daprd: $DEV_DAPRD_BINARY${NC}"
    echo "=" | tr ' ' '=' | head -c 60; echo
    
    check_dev_binaries
    
    # Run daprd directly instead of using the CLI
    echo -e "${BLUE}Starting daprd process...${NC}"
    "$DEV_DAPRD_BINARY" \
        --app-id test-app \
        --components-path ./components \
        --dapr-http-port 3500 \
        --dapr-grpc-port 50001 \
        --log-level info \
        --mode standalone &
    
    local daprd_pid=$!
    echo -e "${GREEN}daprd started with PID: $daprd_pid${NC}"
    
    # Wait for daprd to start
    echo -e "${BLUE}Waiting for daprd to be ready...${NC}"
    local count=0
    while [ $count -lt 30 ]; do
        if curl -s http://localhost:3500/v1.0/healthz > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Dapr sidecar is ready!${NC}"
            break
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    
    if [ $count -eq 30 ]; then
        echo -e "\n${RED}❌ Dapr sidecar failed to start within 30 seconds${NC}"
        kill $daprd_pid 2>/dev/null || true
        exit 1
    fi
    
    # Keep the script running and wait for the process
    echo -e "${GREEN}🚀 Dapr sidecar is running. Press Ctrl+C to stop.${NC}"
    wait $daprd_pid
}

run_with_release_dapr() {
    echo -e "${GREEN}🚀 Starting Dapr Sidecar with Release Binaries${NC}"
    echo "=" | tr ' ' '=' | head -c 50; echo
    
    # Use regular dapr command (from PATH)
    dapr run \
        --app-id test-app \
        --resources-path ./components \
        --dapr-http-port 3500 \
        --dapr-grpc-port 50001 \
        --log-level info \
        sleep 3600
}

cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down Dapr sidecar...${NC}"
    # Kill any running dapr processes for test-app
    pkill -f "test-app" 2>/dev/null || true
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Parse command line arguments
USE_DEV=false
BUILD_DEV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dev)
            USE_DEV=true
            shift
            ;;
        -r|--release)
            USE_DEV=false
            shift
            ;;
        -b|--build)
            BUILD_DEV=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Build if requested (only for dev mode)
if [ "$BUILD_DEV" = true ] && [ "$USE_DEV" = false ]; then
    echo -e "${YELLOW}⚠️  Build flag ignored when not using dev mode${NC}"
fi

if [ "$BUILD_DEV" = true ] && [ "$USE_DEV" = true ]; then
    build_dev_dapr
fi

# Run appropriate version
if [ "$USE_DEV" = true ]; then
    run_with_dev_dapr
else
    run_with_release_dapr
fi

echo -e "${GREEN}🛑 Dapr sidecar stopped${NC}" 