#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        if [ "$2" != "" ]; then
            echo -e "Install with: $2"
        fi
        exit 1
    fi
}

# Clean up function
cleanup_previous_run() {
    echo -e "${BLUE}Cleaning up previous run...${NC}"
    rm -f build/*.dot
    rm -rf visualizations
    mkdir -p visualizations
}

# Check for required commands
check_command cmake "brew install cmake (macOS) or sudo apt-get install cmake (Ubuntu)"
check_command dot "brew install graphviz (macOS) or sudo apt-get install graphviz (Ubuntu)"

# Clean up previous run
cleanup_previous_run

# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd build

# Configure and build project
echo -e "${BLUE}Configuring project...${NC}"
cmake .. -DDEBUG_MODE=OFF

echo -e "${BLUE}Building project...${NC}"
make

# Check if build was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Run the program
echo -e "${BLUE}Running program...${NC}"
if command -v timeout &> /dev/null; then
    # Linux systems with timeout command
    timeout 30s ./CPDL
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo -e "${RED}Program timed out after 30 seconds!${NC}"
        exit 1
    elif [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}Program failed with error code $EXIT_CODE${NC}"
        exit 1
    fi
else
    # macOS or systems without timeout
    ./CPDL &
    PID=$!
    
    # Wait up to 30 seconds
    COUNTER=0
    while kill -0 $PID 2>/dev/null; do
        if [ $COUNTER -ge 30 ]; then
            echo -e "${RED}Program timed out after 30 seconds!${NC}"
            echo -e "${BLUE}Process status before kill:${NC}"
            ps -f $PID
            kill -9 $PID 2>/dev/null
            exit 1
        fi
        sleep 1
        COUNTER=$((COUNTER + 1))
    done
    
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}Program failed with error code $EXIT_CODE${NC}"
        exit 1
    fi
fi

# Generate visualizations
echo -e "${BLUE}Generating visualizations...${NC}"

# Move .dot files to visualizations directory
mv *.dot ../visualizations/ 2>/dev/null || true

# Generate PNGs from DOT files
cd ../visualizations
for dot_file in *.dot; do
    if [ -f "$dot_file" ]; then
        png_file="${dot_file%.dot}.png"
        echo -e "${GREEN}Generating $png_file...${NC}"
        dot -Tpng -Gdpi=300 "$dot_file" -o "$png_file"
        
        # Also generate SVG for vector graphics
        svg_file="${dot_file%.dot}.svg"
        echo -e "${GREEN}Generating $svg_file...${NC}"
        dot -Tsvg "$dot_file" -o "$svg_file"
    fi
done

echo -e "${GREEN}Done! Visualizations are available in the visualizations/ directory${NC}"

# List generated files
echo -e "${BLUE}Generated files:${NC}"
ls -l

cd .. 