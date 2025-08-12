#!/bin/bash
# Multi-task Optuna hyperparameter search script for all datasets
# Bash version for Linux/macOS
# This script runs multi-task Optuna searches for each dataset

# Default parameters
N_TRIALS=20
USE_SUBSET=false
SUBSET_RATIO=0.3
USE_GLOBAL_FEATURES=false
QUICK_MODE=false

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
        --quick-mode)
            QUICK_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --n-trials NUM           Number of Optuna trials per search (default: 20)"
            echo "  --use-subset             Use subset during hyperparameter search"
            echo "  --subset-ratio RATIO     Ratio of dataset to use for subset (default: 0.3)"
            echo "  --use-global-features    Use global molecular features"
            echo "  --quick-mode             Use fewer targets for faster testing"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with default settings"
            echo "  $0 --n-trials 50                     # Run with 50 trials per search"
            echo "  $0 --quick-mode                      # Quick test mode"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "KANG THESIS - MULTI-TASK OPTUNA HYPERPARAMETER SEARCH"
echo "======================================================================"
echo "Starting multi-task Optuna searches..."
echo "Trials per search: $N_TRIALS"
echo "Use subset: $USE_SUBSET"
if [ "$USE_SUBSET" = true ]; then
    echo "Subset ratio: $SUBSET_RATIO"
fi
echo "Use global features: $USE_GLOBAL_FEATURES"
echo "Quick mode: $QUICK_MODE"
echo ""

# Configuration
RESULTS_DIR="experiments/optuna_search"
mkdir -p $RESULTS_DIR

# Get start time
START_TIME=$(date +%s)

# Define datasets and targets (reusing exact arrays from original script)
echo "======================================================================"
echo "DEFINING MULTI-TASK CONFIGURATIONS"
echo "======================================================================"

# QM8 targets (regression)
QM8_TARGETS=("E1-CC2" "E2-CC2" "f1-CC2" "f2-CC2" "E1-PBE0" "E2-PBE0" "f1-PBE0" "f2-PBE0" "E1-CAM" "E2-CAM" "f1-CAM" "f2-CAM")
QM8_QUICK=("E1-CC2" "E2-CC2" "f1-CC2")
if [ "$QUICK_MODE" = true ]; then
    QM8_SELECTED=("${QM8_QUICK[@]}")
else
    QM8_SELECTED=("${QM8_TARGETS[@]}")
fi

# QM9 targets (regression)
QM9_TARGETS=("mu" "alpha" "homo" "lumo" "gap" "r2" "zpve" "u0" "u298" "h298" "g298" "cv")
QM9_QUICK=("mu" "alpha" "homo" "lumo" "gap")
if [ "$QUICK_MODE" = true ]; then
    QM9_SELECTED=("${QM9_QUICK[@]}")
else
    QM9_SELECTED=("${QM9_TARGETS[@]}")
fi

# ToxCast targets (classification)
TOXCAST_ASSAYS=("TOX21_AhR_LUC_Agonist" "TOX21_Aromatase_Inhibition" "TOX21_AutoFluor_HEK293_Cell_blue" "TOX21_p53_BLA_p3_ch1" "TOX21_p53_BLA_p4_ratio")
TOXCAST_QUICK=("TOX21_AhR_LUC_Agonist" "TOX21_Aromatase_Inhibition" "TOX21_AutoFluor_HEK293_Cell_blue")
if [ "$QUICK_MODE" = true ]; then
    TOXCAST_SELECTED=("${TOXCAST_QUICK[@]}")
else
    TOXCAST_SELECTED=("${TOXCAST_ASSAYS[@]}")
fi

TOTAL_TESTS=3  # QM8, QM9, ToxCast

echo "QM8 multi-task regression targets: ${#QM8_SELECTED[@]}"
echo "QM9 multi-task regression targets: ${#QM9_SELECTED[@]}"
echo "ToxCast multi-task classification targets: ${#TOXCAST_SELECTED[@]}"
echo "TOTAL MULTI-TASK SEARCHES TO PERFORM: $TOTAL_TESTS"
echo ""

# Counter for progress tracking
COMPLETED_TESTS=0
FAILED_TESTS=()

# Function to run multi-task Optuna search
run_multitask_optuna_search() {
    local TASK_TYPE=$1
    local DATASET_NAME=$2
    local TARGETS=("${!3}")
    local TEST_NUMBER=$4
    
    echo "======================================================================"
    echo "MULTI-TASK SEARCH ${TEST_NUMBER}/${TOTAL_TESTS}: $TASK_TYPE - $DATASET_NAME"
    echo "Targets (${#TARGETS[@]}): $(IFS=', '; echo "${TARGETS[*]}")"
    echo "Global features: $USE_GLOBAL_FEATURES"
    echo "======================================================================"
    
    # Build command
    CMD="python optuna_search_main.py --task $TASK_TYPE --dataset_name $DATASET_NAME --n_trials $N_TRIALS --multitask"
    
    if [ "$TASK_TYPE" = "regression" ]; then
        TARGETS_STR=$(IFS=','; echo "${TARGETS[*]}")
        CMD="$CMD --multitask_targets $TARGETS_STR"
    else
        TARGETS_STR=$(IFS=','; echo "${TARGETS[*]}")
        CMD="$CMD --multitask_assays $TARGETS_STR"
    fi
    
    if [ "$USE_SUBSET" = true ]; then
        CMD="$CMD --use_subset --subset_ratio $SUBSET_RATIO"
    fi
    
    if [ "$USE_GLOBAL_FEATURES" = true ]; then
        CMD="$CMD --use_global_features"
    fi
    
    echo "Executing: $CMD"
    
    if eval $CMD; then
        echo "SUCCESS: Multi-task $TASK_TYPE - $DATASET_NAME"
    else
        echo "FAILED: Multi-task $TASK_TYPE - $DATASET_NAME"
        FAILED_TESTS+=("Multi-task $TASK_TYPE - $DATASET_NAME")
    fi
    
    ((COMPLETED_TESTS++))
    
    # Progress update
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $TEST_NUMBER -gt 0 ]; then
        AVG_TIME=$((ELAPSED / TEST_NUMBER))
        REMAINING_TIME=$((AVG_TIME * (TOTAL_TESTS - TEST_NUMBER)))
        PROGRESS=$(echo "scale=1; $TEST_NUMBER * 100 / $TOTAL_TESTS" | bc)
        echo "Progress: $TEST_NUMBER/$TOTAL_TESTS - ${PROGRESS}%"
        echo "Elapsed: $(date -d@$ELAPSED -u +%H:%M:%S), Estimated remaining: $(date -d@$REMAINING_TIME -u +%H:%M:%S)"
    fi
    echo ""
}

echo "======================================================================"
echo "RUNNING MULTI-TASK OPTUNA SEARCHES"
echo "======================================================================"

# QM8 multi-task regression
run_multitask_optuna_search "regression" "QM8" QM8_SELECTED[@] 1

# QM9 multi-task regression
run_multitask_optuna_search "regression" "QM9" QM9_SELECTED[@] 2

# ToxCast multi-task classification
run_multitask_optuna_search "classification" "TOXCAST" TOXCAST_SELECTED[@] 3

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "======================================================================"
echo "MULTI-TASK OPTUNA SEARCH COMPLETED!"
echo "======================================================================"
echo "Total execution time: $(date -d@$TOTAL_TIME -u +%H:%M:%S)"
echo "Completed searches: $((TOTAL_TESTS - ${#FAILED_TESTS[@]}))/$TOTAL_TESTS"
if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "Failed searches: ${#FAILED_TESTS[@]}"
    echo "Failed tests:"
    for failed in "${FAILED_TESTS[@]}"; do
        echo "  - $failed"
    done
fi
echo ""
echo "Results saved in: experiments/optuna_search/"
echo "Use python analyze_optuna_results.py to analyze results"
