#!/bin/bash
# =============================================================================
# run_compare.sh
# Run BOTH strategies (reference baseline + our optimized alternative)
# and compute speedup for each dataset size.
# =============================================================================

set -e

REF_BIN="./nn_reference"
OPT_BIN="./nn_optimized"
DATA_DIR="./data"
RESULTS_DIR="./results"
COMPARE_FILE="$RESULTS_DIR/comparison.csv"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# 1. Build both if needed
# ---------------------------------------------------------------------------
if [ ! -f "$REF_BIN" ]; then
    echo -e "${YELLOW}Reference binary not found — building...${NC}"
    make reference
fi

if [ ! -f "$OPT_BIN" ]; then
    echo -e "${YELLOW}Optimized binary not found — building...${NC}"
    make optimized
fi

# ---------------------------------------------------------------------------
# 2. Check data
# ---------------------------------------------------------------------------
for size in small medium large; do
    if [ ! -f "$DATA_DIR/synthetic_convex_$size.csv" ]; then
        echo "ERROR: Missing $DATA_DIR/synthetic_convex_$size.csv"
        echo "Run: python3 playground/generate_synthetic_data.py"
        exit 1
    fi
done

mkdir -p "$RESULTS_DIR"
echo "dataset,reference_time_s,optimized_time_s,speedup" > "$COMPARE_FILE"

# ---------------------------------------------------------------------------
# 3. Benchmark both strategies
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}${CYAN}================================================================"
echo -e " CUDA Neural Network — Strategy Comparison"
echo -e " Reference:  naive mat_mult (global memory only)"
echo -e " Optimized:  tiling + transpose + fused forward pass"
echo -e "================================================================${NC}\n"

declare -A REF_TIMES
declare -A OPT_TIMES

for size in small medium large; do
    DATA="$DATA_DIR/synthetic_convex_$size.csv"

    echo -e "${YELLOW}──── Dataset: $size ────${NC}"

    # --- Reference ---
    echo -e "  ${RED}▶ Reference baseline...${NC}"
    REF_OUT=$("$REF_BIN" "$DATA" 2>&1)
    REF_TIME=$(echo "$REF_OUT" | grep "Training time" | awk '{print $3}')
    REF_TIMES[$size]=$REF_TIME
    echo "    Time: ${REF_TIME}s"

    # --- Optimized ---
    echo -e "  ${GREEN}▶ Optimized alternative...${NC}"
    OPT_OUT=$("$OPT_BIN" "$DATA" 2>&1)
    OPT_TIME=$(echo "$OPT_OUT" | grep "Training time" | awk '{print $3}')
    OPT_TIMES[$size]=$OPT_TIME
    echo "    Time: ${OPT_TIME}s"

    # Compute speedup
    SPEEDUP=$(python3 -c "
ref=$REF_TIME; opt=$OPT_TIME
if opt > 0:
    print(f'{ref/opt:.2f}')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")

    echo -e "  ${BOLD}Speedup: ${SPEEDUP}x${NC}"
    echo "$size,$REF_TIME,$OPT_TIME,$SPEEDUP" >> "$COMPARE_FILE"
    echo ""
done

# ---------------------------------------------------------------------------
# 4. Summary table
# ---------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}================================================================"
echo -e " Results Summary"
echo -e "================================================================${NC}"
printf "%-10s %-18s %-18s %-10s\n" "Dataset" "Reference (s)" "Optimized (s)" "Speedup"
printf "%-10s %-18s %-18s %-10s\n" "-------" "-------------" "-------------" "-------"

while IFS=, read -r dataset ref opt speedup; do
    [ "$dataset" = "dataset" ] && continue
    printf "%-10s %-18s %-18s %-10s\n" "$dataset" "$ref" "$opt" "${speedup}x"
done < "$COMPARE_FILE"

echo ""
echo -e "Full results saved to: ${BOLD}$COMPARE_FILE${NC}"