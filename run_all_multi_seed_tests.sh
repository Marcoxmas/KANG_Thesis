#!/bin/bash

# Comprehensive multi-seed testing script for all datasets and columns
# This script automatically runs multi-seed tests for every dataset/column combination
# and generates comprehensive summaries

set -e  # Exit on any error

# Default parameters
NUM_SEEDS=5
EPOCHS=50
PATIENCE=25
FIRST_TARGET_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --first-target-only)
            FIRST_TARGET_ONLY=true
            shift
            ;;
        --num-seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --first-target-only    Test only the first target from each dataset"
            echo "  --num-seeds NUM        Number of seeds to test (default: 5)"
            echo "  --epochs NUM           Number of epochs (default: 50)"
            echo "  --patience NUM         Patience for early stopping (default: 25)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "KANG THESIS - COMPREHENSIVE MULTI-SEED TESTING SUITE"
echo "======================================================================"
echo "Starting comprehensive multi-seed testing across all datasets..."
if [ "$FIRST_TARGET_ONLY" = true ]; then
    echo "QUICK TEST MODE: Testing only the first target from each dataset"
fi
echo ""

# Configuration
RESULTS_DIR="experiments/multi_seed_results"
SUMMARY_DIR="experiments/multi_seed_summaries"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$SUMMARY_DIR"

# Counter for tracking progress
total_tests=0
completed_tests=0

echo "======================================================================"
echo "PHASE 1: COUNTING TOTAL TESTS TO PERFORM"
echo "======================================================================"

# QM8 targets (regression)
qm8_targets=("E1-CC2" "E2-CC2" "f1-CC2" "f2-CC2" "E1-PBE0" "E2-PBE0" "f1-PBE0" "f2-PBE0" "E1-CAM" "E2-CAM" "f1-CAM" "f2-CAM")
if [ "$FIRST_TARGET_ONLY" = true ]; then
    qm8_targets=("${qm8_targets[0]}")  # Just E1-CC2
fi
total_tests=$((total_tests + ${#qm8_targets[@]}))
echo "QM8 regression targets: ${#qm8_targets[@]}"

# QM9 targets (regression)
qm9_targets=("mu" "alpha" "homo" "lumo" "gap" "r2" "zpve" "u0" "u298" "h298" "g298" "cv")
if [ "$FIRST_TARGET_ONLY" = true ]; then
    qm9_targets=("${qm9_targets[0]}")  # Just mu
fi
total_tests=$((total_tests + ${#qm9_targets[@]}))
echo "QM9 regression targets: ${#qm9_targets[@]}"

# HIV (classification)
total_tests=$((total_tests + 1))
echo "HIV classification: 1"

# ToxCast targets (classification) - Focus on specific 5 assays
echo "Using predefined ToxCast assays for thesis..."

# Define the 5 specific ToxCast assays for the thesis
toxcast_assays=(
    "TOX21_AhR_LUC_Agonist"
    "TOX21_Aromatase_Inhibition" 
    "TOX21_AutoFluor_HEK293_Cell_blue"
    "TOX21_p53_BLA_p3_ch1"
    "TOX21_p53_BLA_p4_ratio"
)

if [ "$FIRST_TARGET_ONLY" = true ]; then
    toxcast_assays=("${toxcast_assays[0]}")  # Just TOX21_AhR_LUC_Agonist
fi

echo "Selected ToxCast assays for thesis:"
for i in "${!toxcast_assays[@]}"; do
    echo "$((i+1)). ${toxcast_assays[i]}"
done

total_tests=$((total_tests + ${#toxcast_assays[@]}))
echo "ToxCast classification targets: ${#toxcast_assays[@]}"

if [ "$FIRST_TARGET_ONLY" = true ]; then
    echo ""
    echo "FIRST TARGET ONLY MODE - Selected targets:"
    echo "  QM8: ${qm8_targets[0]}"
    echo "  QM9: ${qm9_targets[0]}"
    echo "  HIV: HIV (classification)"
    echo "  ToxCast: ${toxcast_assays[0]}"
fi

echo ""
echo "TOTAL TESTS TO PERFORM: $total_tests"
echo "Estimated time: $((total_tests * 15)) minutes (assuming 15 min per test)"
echo ""

# Function to run a single test and handle errors
run_single_test() {
    local task_type="$1"
    local dataset_name="$2"
    local target_column="$3"
    local test_name="$4"
    
    echo "[$((completed_tests + 1))/$total_tests] Running: $test_name"
    echo "  Task: $task_type | Dataset: $dataset_name | Target: $target_column"
    
    # Create a unique log file for this test
    local log_file="$RESULTS_DIR/${task_type}_${dataset_name}"
    if [ -n "$target_column" ]; then
        log_file="${log_file}_${target_column}"
    fi
    log_file="${log_file}_${NUM_SEEDS}seeds.log"
    
    # Run the test with error handling
    if [ -n "$target_column" ]; then
        if python3 multi_seed_test.py --task "$task_type" --dataset_name "$dataset_name" --target_column "$target_column" --num_seeds "$NUM_SEEDS" --epochs "$EPOCHS" --patience "$PATIENCE" > "$log_file" 2>&1; then
            echo "  ✓ SUCCESS"
        else
            echo "  ✗ FAILED - Check $log_file for details"
            echo "FAILED: $test_name" >> "$RESULTS_DIR/failed_tests.txt"
        fi
    else
        if python3 multi_seed_test.py --task "$task_type" --dataset_name "$dataset_name" --num_seeds "$NUM_SEEDS" --epochs "$EPOCHS" --patience "$PATIENCE" > "$log_file" 2>&1; then
            echo "  ✓ SUCCESS"
        else
            echo "  ✗ FAILED - Check $log_file for details"
            echo "FAILED: $test_name" >> "$RESULTS_DIR/failed_tests.txt"
        fi
    fi
    
    completed_tests=$((completed_tests + 1))
    echo "  Progress: $completed_tests/$total_tests completed"
    echo ""
}

echo "======================================================================"
echo "PHASE 2: RUNNING MULTI-SEED TESTS"
echo "======================================================================"

# Initialize failed tests file
echo "# Failed Tests Log" > "$RESULTS_DIR/failed_tests.txt"
echo "# Generated on $(date)" >> "$RESULTS_DIR/failed_tests.txt"
echo "" >> "$RESULTS_DIR/failed_tests.txt"

start_time=$(date +%s)

echo ""
echo "--- QM8 REGRESSION TESTS ---"
for target in "${qm8_targets[@]}"; do
    run_single_test "regression" "QM8" "$target" "QM8-$target"
done

echo ""
echo "--- QM9 REGRESSION TESTS ---"
for target in "${qm9_targets[@]}"; do
    run_single_test "regression" "QM9" "$target" "QM9-$target"
done

echo ""
echo "--- HIV CLASSIFICATION TEST ---"
run_single_test "classification" "HIV" "" "HIV"

echo ""
echo "--- TOXCAST CLASSIFICATION TESTS ---"
for assay in "${toxcast_assays[@]}"; do
    run_single_test "classification" "$assay" "" "ToxCast-$assay"
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "======================================================================"
echo "PHASE 3: GENERATING COMPREHENSIVE SUMMARIES"
echo "======================================================================"

echo "Running comprehensive summary generation..."
python3 generate_comprehensive_summary.py

echo "======================================================================"
echo "COMPREHENSIVE MULTI-SEED TESTING COMPLETE!"
echo "======================================================================"

echo ""
echo "EXECUTION SUMMARY:"
echo "  Total tests attempted: $total_tests"
echo "  Total tests completed: $completed_tests"
echo "  Total execution time: $((total_time / 3600))h $((total_time % 3600 / 60))m $((total_time % 60))s"
echo ""
echo "RESULTS LOCATIONS:"
echo "  Individual results: $RESULTS_DIR/"
echo "  Comprehensive summaries: $SUMMARY_DIR/"
echo "  Test logs: $RESULTS_DIR/*.log"
echo ""

# Check for failed tests
if [ -f "$RESULTS_DIR/failed_tests.txt" ]; then
    failed_count=$(grep -c "FAILED:" "$RESULTS_DIR/failed_tests.txt" 2>/dev/null || echo "0")
    if [ "$failed_count" -gt 0 ]; then
        echo "WARNING: $failed_count tests failed. Check $RESULTS_DIR/failed_tests.txt for details."
    fi
fi

echo "To view the main summary:"
echo "  cat $SUMMARY_DIR/comprehensive_summary.txt"
echo ""
echo "To view performance plots:"
echo "  Open $SUMMARY_DIR/performance_overview.png"
echo "  Open $SUMMARY_DIR/global_features_impact.png"
echo ""
echo "To view LaTeX table:"
echo "  cat $SUMMARY_DIR/latex_summary_table.txt"
echo ""

echo "Script execution completed successfully!"
