#!/bin/bash

# TOXCAST Training Script - Specialized for TOXCAST dataset configurations
# Works with parameter files in format: best_params_{task_type}_TOXCAST_{target}_global_{bool}_3d_{bool}_{loops}.json
# Colors

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
EPOCHS=200
PATIENCE=50
SEED=42
DRY_RUN=false

# Configuration variables
TASK_TYPE=""
HEAD_TYPE=""
GLOBAL_FEATURES=""
USE_3D=""
SELF_LOOPS=""

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required Configuration:"
    echo "  --single-task        Run all single-task TOXCAST targets"
    echo "  --multitask          Run TOXCAST multitask configurations"
    echo ""
    echo "Feature Configuration:"
    echo "  --global             Use global features (global_True)"
    echo "  --no-global          Don't use global features (global_False)"
    echo "  --3d                 Use 3D geometry (3d_True)"
    echo "  --no-3d              Don't use 3D geometry (3d_False)"
    echo "  --loops              Use self loops (with_loops)"
    echo "  --no-loops           Don't use self loops (no_loops)"
    echo ""
    echo "Multitask Options:"
    echo "  --single-head        Use single head (for multitask only)"
    echo "  --multi-head         Use multi head (for multitask only)"
    echo ""
    echo "Training Parameters:"
    echo "  --epochs N           Number of epochs (default: 200)"
    echo "  --patience N         Early stopping patience (default: 50)"
    echo "  --seed N             Random seed (default: 42)"
    echo ""
    echo "Other:"
    echo "  --dry-run           Show commands without executing"
    echo "  --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --single-task --no-global --no-3d --loops"
    echo "  $0 --multitask --global --3d --no-loops --single-head"
    echo "  $0 --single-task --global --no-3d --loops --epochs 300 --seed 123"
}

log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${BLUE}[INFO]${NC}  $timestamp - $message" ;;
        "OK")    echo -e "${GREEN}[OK]${NC}    $timestamp - $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  $timestamp - $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $timestamp - $message" ;;
    esac
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --single-task)
            TASK_TYPE="single"
            shift ;;
        --multitask)
            TASK_TYPE="multi"
            shift ;;
        --single-head)
            HEAD_TYPE="singlehead"
            shift ;;
        --multi-head)
            HEAD_TYPE="multihead"
            shift ;;
        --global)
            GLOBAL_FEATURES="True"
            shift ;;
        --no-global)
            GLOBAL_FEATURES="False"
            shift ;;
        --3d)
            USE_3D="True"
            shift ;;
        --no-3d)
            USE_3D="False"
            shift ;;
        --loops)
            SELF_LOOPS="with_loops"
            shift ;;
        --no-loops)
            SELF_LOOPS="no_loops"
            shift ;;
        --epochs)
            EPOCHS="$2"
            shift 2 ;;
        --patience)
            PATIENCE="$2"
            shift 2 ;;
        --seed)
            SEED="$2"
            shift 2 ;;
        --dry-run)
            DRY_RUN=true
            shift ;;
        --help)
            print_usage
            exit 0 ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1 ;;
    esac
done

# Validate required arguments
if [[ -z "$TASK_TYPE" ]]; then
    echo -e "${RED}Error: Must specify either --single-task or --multitask${NC}"
    print_usage
    exit 1
fi

if [[ -z "$GLOBAL_FEATURES" || -z "$USE_3D" || -z "$SELF_LOOPS" ]]; then
    echo -e "${RED}Error: Must specify all feature flags (--global/--no-global, --3d/--no-3d, --loops/--no-loops)${NC}"
    print_usage
    exit 1
fi

if [[ "$TASK_TYPE" == "multi" && -z "$HEAD_TYPE" ]]; then
    echo -e "${RED}Error: Must specify --single-head or --multi-head for multitask${NC}"
    print_usage
    exit 1
fi

# Show configuration
log "INFO" "TOXCAST Training Configuration:"
log "INFO" "  Task: $TASK_TYPE"
if [[ "$TASK_TYPE" == "multi" ]]; then
    log "INFO" "  Head: $HEAD_TYPE"
fi
log "INFO" "  Global features: $GLOBAL_FEATURES"
log "INFO" "  3D geometry: $USE_3D"
log "INFO" "  Self loops: $SELF_LOOPS"
log "INFO" "  Epochs: $EPOCHS, Patience: $PATIENCE, Seed: $SEED"

if [[ "$DRY_RUN" == "true" ]]; then
    log "WARN" "DRY RUN MODE - No training will be executed"
fi

# Find matching TOXCAST parameter files
log "INFO" "Searching for TOXCAST parameter files..."

optuna_dir="experiments/optuna_search"
if [[ ! -d "$optuna_dir" ]]; then
    log "ERROR" "Directory not found: $optuna_dir"
    exit 1
fi

# Build search pattern for TOXCAST specifically
if [[ "$TASK_TYPE" == "single" ]]; then
    # Explicitly exclude multitask files for single task
    pattern="best_params_*_TOXCAST_*_global_${GLOBAL_FEATURES}_3d_${USE_3D}_${SELF_LOOPS}.json"
    # We'll filter out multitask in the loop below
else
    pattern="best_params_*_TOXCAST_${HEAD_TYPE}_multitask_*_global_${GLOBAL_FEATURES}_3d_${USE_3D}_${SELF_LOOPS}.json"
fi

log "INFO" "Pattern: $pattern"

# Find matching files with explicit filtering
matching_files=()
total_count=0

if [[ "$TASK_TYPE" == "single" ]]; then
    # For single task: find all files matching pattern, then exclude multitask
    for file in $optuna_dir/$pattern; do
        if [[ -f "$file" ]]; then
            filename=$(basename "$file")
            # Skip multitask files
            if [[ "$filename" == *"multitask"* ]]; then
                continue
            fi
            matching_files+=("$file")
            ((total_count++))
        fi
    done
else
    # For multitask: pattern already includes multitask
    for file in $optuna_dir/$pattern; do
        if [[ -f "$file" ]]; then
            matching_files+=("$file")
            ((total_count++))
        fi
    done
fi

if [[ $total_count -eq 0 ]]; then
    log "ERROR" "No TOXCAST parameter files found matching:"
    log "ERROR" "  Pattern: $pattern"
    log "ERROR" "Available TOXCAST files in $optuna_dir:"
    ls -1 "$optuna_dir"/best_params_*_TOXCAST_*.json 2>/dev/null | head -10 || log "ERROR" "No TOXCAST files found"
    exit 1
fi

log "OK" "Found $total_count matching TOXCAST parameter files"

# Extract unique TOXCAST targets
toxcast_targets=()
run_count=0

for file in "${matching_files[@]}"; do
    filename=$(basename "$file")
    
    # Extract target name from TOXCAST files
    if [[ "$TASK_TYPE" == "single" ]]; then
        # For single task: best_params_{task_type}_TOXCAST_{target}_global_{bool}_3d_{bool}_{loops}.json
        if [[ "$filename" =~ best_params_(classification|regression)_TOXCAST_(.+)_global_${GLOBAL_FEATURES}_3d_${USE_3D}_${SELF_LOOPS}\.json$ ]]; then
            task_type_from_file="${BASH_REMATCH[1]}"
            target="${BASH_REMATCH[2]}"
            
            # Add target to list if not already there
            found=false
            for t in "${toxcast_targets[@]}"; do
                if [[ "$t" == "$target" ]]; then
                    found=true
                    break
                fi
            done
            if [[ "$found" == "false" ]]; then
                toxcast_targets+=("$target")
            fi
            
            ((run_count++))
        fi
    else
        # For multitask: best_params_{task_type}_TOXCAST_{head_type}_multitask_*_global_{bool}_3d_{bool}_{loops}.json
        if [[ "$filename" =~ best_params_(classification|regression)_TOXCAST_${HEAD_TYPE}_multitask_ ]]; then
            task_type_from_file="${BASH_REMATCH[1]}"
            
            # For multitask, we don't have individual targets, just mark as multitask
            found=false
            for t in "${toxcast_targets[@]}"; do
                if [[ "$t" == "multitask" ]]; then
                    found=true
                    break
                fi
            done
            if [[ "$found" == "false" ]]; then
                toxcast_targets+=("multitask")
            fi
            
            ((run_count++))
        fi
    fi
done

if [[ "$TASK_TYPE" == "single" ]]; then
    log "INFO" "Will run training for TOXCAST targets: ${toxcast_targets[*]}"
else
    log "INFO" "Will run TOXCAST multitask training"
fi
log "INFO" "Total training runs: $run_count"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    log "INFO" "Commands that would be executed:"
fi

# Execute training runs
successful=0
failed=0

mkdir -p experiments/training_results

for file in "${matching_files[@]}"; do
    filename=$(basename "$file")
    
    # Parse filename to extract information
    if [[ "$TASK_TYPE" == "single" ]]; then
        # Single task: best_params_{task_type}_TOXCAST_{target}_global_{bool}_3d_{bool}_{loops}.json
        # Note: target can contain underscores, so we need to match everything between TOXCAST and _global_
        if [[ "$filename" =~ best_params_(classification|regression)_TOXCAST_(.+)_global_${GLOBAL_FEATURES}_3d_${USE_3D}_${SELF_LOOPS}\.json$ ]]; then
            target="${BASH_REMATCH[2]}"
            
            cmd="python -u train_with_best_params.py --dataset_name TOXCAST --target_column $target --epochs $EPOCHS --patience $PATIENCE --seed $SEED"
            
            # Add feature flags
            if [[ "$GLOBAL_FEATURES" == "True" ]]; then
                cmd="$cmd --use_global_features"
            fi
            if [[ "$USE_3D" == "True" ]]; then
                cmd="$cmd --use_3d_geo"
            fi
            if [[ "$SELF_LOOPS" == "no_loops" ]]; then
                cmd="$cmd --no_self_loops"
            fi
            
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "  $cmd"
            else
                log "INFO" "Training TOXCAST - $target"
                if eval "$cmd"; then
                    log "OK" "Success: TOXCAST - $target"
                    ((successful++))
                else
                    log "ERROR" "Failed: TOXCAST - $target"
                    ((failed++))
                fi
            fi
        fi
        
    else
        # Multitask: best_params_{task_type}_TOXCAST_{head_type}_multitask_*_global_{bool}_3d_{bool}_{loops}.json
        if [[ "$filename" =~ best_params_(classification|regression)_TOXCAST_${HEAD_TYPE}_multitask_ ]]; then
            
            cmd="python -u train_with_best_params.py --dataset_name TOXCAST --multitask --epochs $EPOCHS --patience $PATIENCE --seed $SEED"
            
            if [[ "$HEAD_TYPE" == "singlehead" ]]; then
                cmd="$cmd --single_head"
            fi
            
            # Add feature flags
            if [[ "$GLOBAL_FEATURES" == "True" ]]; then
                cmd="$cmd --use_global_features"
            fi
            if [[ "$USE_3D" == "True" ]]; then
                cmd="$cmd --use_3d_geo"
            fi
            if [[ "$SELF_LOOPS" == "no_loops" ]]; then
                cmd="$cmd --no_self_loops"
            fi
            
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "  $cmd"
            else
                log "INFO" "Training TOXCAST - multitask ($HEAD_TYPE)"
                if eval "$cmd"; then
                    log "OK" "Success: TOXCAST - multitask ($HEAD_TYPE)"
                    ((successful++))
                else
                    log "ERROR" "Failed: TOXCAST - multitask ($HEAD_TYPE)"
                    ((failed++))
                fi
            fi
        fi
    fi
done

# Final summary
if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    log "INFO" "TOXCAST training completed!"
    log "INFO" "Results: $successful successful, $failed failed"
    
    if [[ $failed -gt 0 ]]; then
        exit 1
    fi
fi

log "OK" "Done!"