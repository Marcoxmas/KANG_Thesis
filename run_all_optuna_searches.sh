#!/bin/bash

# Comprehensive Optuna hyperparameter search script for all datasets
# Bash version for Unix/Linux/macOS
# This script automatically runs Optuna searches for every dataset/target combination

set -e  # Exit on any error

# Default parameters
N_TRIALS=20
USE_SUBSET=false
SUBSET_RATIO=0.3
USE_GLOBAL_FEATURES=false
FIRST_TARGET_ONLY=false

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --n-trials NUM         Number of Optuna trials per search (default: 20)"
    echo "  --use-subset           Use subset during hyperparameter search"
    echo "  --subset-ratio RATIO   Ratio of dataset to use for subset (default: 0.3)"
    echo "  --use-global-features  Use global molecular features"
    echo "  --first-target-only    Test only the first target from each dataset (for quick testing)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                # Run with default settings"
    echo "  $0 --n-trials 50                 # Run with 50 trials per search"
    echo "  $0 --use-global-features         # Use global features"
    echo "  $0 --first-target-only           # Quick test mode"
    echo "  $0 --use-subset --subset-ratio 0.2  # Use 20% subset during search"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --use-subset)
            USE_SUBSET=true
            shift
            ;;
        --subset-ratio)
            SUBSET_RATIO="$2"
            shift 2
            ;;
        --use-global-features)
            USE_GLOBAL_FEATURES=true
            shift
            ;;
        --first-target-only)
            FIRST_TARGET_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "KANG THESIS - COMPREHENSIVE OPTUNA HYPERPARAMETER SEARCH"
echo "======================================================================"
echo "Starting comprehensive Optuna search across all datasets..."
echo "Trials per search: $N_TRIALS"
echo "Use subset: $USE_SUBSET"
if [ "$USE_SUBSET" = true ]; then
    echo "Subset ratio: $SUBSET_RATIO"
fi
echo "Use global features: $USE_GLOBAL_FEATURES"
if [ "$FIRST_TARGET_ONLY" = true ]; then
    echo "QUICK TEST MODE: Testing only the first target from each dataset"
fi
echo ""

# Configuration
RESULTS_DIR="experiments/optuna_search"

# Create directories
mkdir -p "$RESULTS_DIR"

# Get start time
start_time=$(date +%s)

# Define all datasets and targets
echo "======================================================================"
echo "DEFINING TEST CONFIGURATIONS"
echo "======================================================================"

# QM8 targets (regression)
if [ "$FIRST_TARGET_ONLY" = true ]; then
    qm8_targets=("E1-CC2")
else
    qm8_targets=("E1-CC2" "E2-CC2" "f1-CC2" "f2-CC2" "E1-PBE0" "E2-PBE0" "f1-PBE0" "f2-PBE0" "E1-CAM" "E2-CAM" "f1-CAM" "f2-CAM")
fi

# QM9 targets (regression)
if [ "$FIRST_TARGET_ONLY" = true ]; then
    qm9_targets=("mu")
else
    qm9_targets=("mu" "alpha" "homo" "lumo" "gap" "r2" "zpve" "u0" "u298" "h298" "g298" "cv")
fi

# ToxCast targets (classification)
if [ "$FIRST_TARGET_ONLY" = true ]; then
    toxcast_assays=("TOX21_AhR_LUC_Agonist")
else
    toxcast_assays=(
        "TOX21_AhR_LUC_Agonist"
        "TOX21_Aromatase_Inhibition"
        "TOX21_AutoFluor_HEK293_Cell_blue"
        "TOX21_p53_BLA_p3_ch1"
        "TOX21_p53_BLA_p4_ratio"
    )
fi

total_tests=$((${#qm8_targets[@]} + ${#qm9_targets[@]} + 1 + ${#toxcast_assays[@]}))  # +1 for HIV

echo "QM8 regression targets: ${#qm8_targets[@]}"
echo "QM9 regression targets: ${#qm9_targets[@]}"
echo "HIV classification: 1"
echo "ToxCast classification targets: ${#toxcast_assays[@]}"
echo "TOTAL SEARCHES TO PERFORM: $total_tests"
echo "Estimated time: $((total_tests * 30)) minutes (assuming 30 min per search)"
echo ""

# Counter for progress tracking
completed_tests=0
failed_tests=()

# Function to format time duration
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local seconds=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# Function to run individual Optuna search
run_optuna_search() {
    local task_type="$1"
    local dataset_name="$2"
    local target_column="$3"
    local test_number="$4"
    
    echo "======================================================================"
    echo "SEARCH $test_number/$total_tests: $task_type - $dataset_name"
    if [ -n "$target_column" ]; then
        echo "Target: $target_column"
    fi
    echo "Global features: $USE_GLOBAL_FEATURES"
    echo "======================================================================"
    
    # Build command
    cmd="python optuna_search_main.py --task $task_type --dataset_name $dataset_name --n_trials $N_TRIALS"
    
    if [ -n "$target_column" ]; then
        cmd="$cmd --target_column $target_column"
    fi
    
    if [ "$USE_SUBSET" = true ]; then
        cmd="$cmd --use_subset --subset_ratio $SUBSET_RATIO"
    fi
    
    if [ "$USE_GLOBAL_FEATURES" = true ]; then
        cmd="$cmd --use_global_features"
    fi
    
    echo "Executing: $cmd"
    
    if $cmd; then
        echo "✓ SUCCESS: $task_type - $dataset_name"
    else
        echo "✗ FAILED: $task_type - $dataset_name (Exit code: $?)"
        failed_tests+=("$task_type - $dataset_name - $target_column")
    fi
    
    ((completed_tests++))
    
    # Progress update
    progress=$(echo "scale=1; $test_number * 100 / $total_tests" | bc -l)
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $test_number -gt 0 ]; then
        avg_time=$((elapsed / test_number))
        remaining_time=$((avg_time * (total_tests - test_number)))
        echo "Progress: $test_number/$total_tests ($progress%)"
        echo "Elapsed: $(format_time $elapsed), Estimated remaining: $(format_time $remaining_time)"
    fi
    echo ""
}

echo "======================================================================"
echo "RUNNING OPTUNA SEARCHES"
echo "======================================================================"

test_number=0

# QM8 regression
for target in "${qm8_targets[@]}"; do
    ((test_number++))
    run_optuna_search "regression" "QM8" "$target" "$test_number"
done

# QM9 regression  
for target in "${qm9_targets[@]}"; do
    ((test_number++))
    run_optuna_search "regression" "QM9" "$target" "$test_number"
done

# HIV classification
((test_number++))
run_optuna_search "classification" "HIV" "" "$test_number"

# ToxCast classification
for assay in "${toxcast_assays[@]}"; do
    ((test_number++))
    run_optuna_search "classification" "TOXCAST" "$assay" "$test_number"
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "======================================================================"
echo "OPTUNA SEARCH COMPLETED!"
echo "======================================================================"
echo "Total execution time: $(format_time $total_time)"
echo "Completed searches: $((total_tests - ${#failed_tests[@]}))/$total_tests"
if [ ${#failed_tests[@]} -gt 0 ]; then
    echo "Failed searches: ${#failed_tests[@]}"
    echo "Failed tests:"
    for failed in "${failed_tests[@]}"; do
        echo "  - $failed"
    done
fi
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo "Use 'python analyze_optuna_results.py' to analyze results"
