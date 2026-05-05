#!/bin/bash
# =============================================================================
# run_optimized.sh
# Run our alternative CUDA strategy (tiling + transpose + kernel fusion)
# on all three datasets and save results to results/optimized.csv
# =============================================================================

set -e

BINARY="./nn_optimized"
DATA_DIR="./data"
RESULTS_DIR="./results"
RESULTS_FILE="$RESULTS_DIR/optimized.csv"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# 1. Build if binary doesn't exist
# ---------------------------------------------------------------------------
if [ ! -f "$BINARY" ]; then
    echo -e "${YELLOW}Binary not found — building...${NC}"
    make optimized
fi

# ---------------------------------------------------------------------------
# 2. Check data exists
# ---------------------------------------------------------------------------
for size in small medium large; do
    if [ ! -f "$DATA_DIR/synthetic_convex_$size.csv" ]; then
        echo "ERROR: $DATA_DIR/synthetic_convex_$size.csv not found."
        echo "Run: python3 playground/generate_synthetic_data.py"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 3. Run on all datasets
# ---------------------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
echo "dataset,time_seconds" > "$RESULTS_FILE"

echo -e "\n${CYAN}=== Optimized Strategy: Tiling + Transpose + Kernel Fusion ===${NC}"
echo -e "Optimizations: Tiled shared memory | Pre-transposed B | Fused forward pass\n"

for size in small medium large; do
    DATA="$DATA_DIR/synthetic_convex_$size.csv"
    echo -e "${GREEN}▶ Dataset: $size${NC}  ($DATA)"

    # Capture output and extract training time
    OUTPUT=$("$BINARY" "$DATA" 2>&1)
    echo "$OUTPUT"

    # Extract time from output line "Training time: X.XXXX seconds"
    TIME=$(echo "$OUTPUT" | grep "Training time" | awk '{print $3}')
    echo "$size,$TIME" >> "$RESULTS_FILE"
    echo ""
done

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
echo -e "${CYAN}=== Results saved to $RESULTS_FILE ===${NC}"
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Run ./run_compare.sh to compare against the reference baseline."