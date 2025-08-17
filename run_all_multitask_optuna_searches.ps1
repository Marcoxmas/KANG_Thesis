# Multi-task Optuna hyperparameter search script for all datasets
# PowerShell version for Windows
# This script runs multi-task Optuna searches for each dataset

param(
    [int]$NTrials = 20,
    [switch]$UseSubset = $false,
    [double]$SubsetRatio = 0.3,
    [switch]$UseGlobalFeatures = $false,
    [switch]$QuickMode = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "Usage: .\run_all_multitask_optuna_searches.ps1 [OPTIONS]" -ForegroundColor Green
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -NTrials NUM           Number of Optuna trials per search (default: 20)" -ForegroundColor White
    Write-Host "  -UseSubset             Use subset during hyperparameter search" -ForegroundColor White
    Write-Host "  -SubsetRatio RATIO     Ratio of dataset to use for subset (default: 0.3)" -ForegroundColor White
    Write-Host "  -UseGlobalFeatures     Use global molecular features" -ForegroundColor White
    Write-Host "  -QuickMode             Use fewer targets for faster testing" -ForegroundColor White
    Write-Host "  -Help                  Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_all_multitask_optuna_searches.ps1                    # Run with default settings" -ForegroundColor White
    Write-Host "  .\run_all_multitask_optuna_searches.ps1 -NTrials 50       # Run with 50 trials per search" -ForegroundColor White
    Write-Host "  .\run_all_multitask_optuna_searches.ps1 -QuickMode        # Quick test mode" -ForegroundColor White
    exit 0
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "KANG THESIS - MULTI-TASK OPTUNA HYPERPARAMETER SEARCH" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting multi-task Optuna searches..." -ForegroundColor Green
Write-Host "Trials per search: $NTrials" -ForegroundColor White
Write-Host "Use subset: $UseSubset" -ForegroundColor White
if ($UseSubset) {
    Write-Host "Subset ratio: $SubsetRatio" -ForegroundColor White
}
Write-Host "Use global features: $UseGlobalFeatures" -ForegroundColor White
Write-Host "Quick mode: $QuickMode" -ForegroundColor White
Write-Host ""

# Configuration
$RESULTS_DIR = "experiments/optuna_search"
New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null

# Get start time
$startTime = Get-Date

# Define datasets and targets (reusing exact arrays from original script)
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "DEFINING MULTI-TASK CONFIGURATIONS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# QM8 targets (regression)
$qm8Targets = @("E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM")
$qm8Quick = @("E1-CC2", "E2-CC2", "f1-CC2")
$qm8Selected = if ($QuickMode) { $qm8Quick } else { $qm8Targets }

# QM9 targets (regression)
$qm9Targets = @("mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv")
$qm9Quick = @("mu", "alpha", "homo", "lumo", "gap")
$qm9Selected = if ($QuickMode) { $qm9Quick } else { $qm9Targets }

# ToxCast targets (classification)
$toxcastAssays = @(
    "TOX21_AhR_LUC_Agonist",
    "TOX21_Aromatase_Inhibition", 
    "TOX21_AutoFluor_HEK293_Cell_blue",
    "TOX21_p53_BLA_p3_ch1",
    "TOX21_p53_BLA_p4_ratio"
)
$toxcastQuick = @("TOX21_AhR_LUC_Agonist", "TOX21_Aromatase_Inhibition", "TOX21_AutoFluor_HEK293_Cell_blue")
$toxcastSelected = if ($QuickMode) { $toxcastQuick } else { $toxcastAssays }

$totalTests = 3  # QM8, QM9, ToxCast

Write-Host "QM8 multi-task regression targets: $($qm8Selected.Count)" -ForegroundColor White
Write-Host "QM9 multi-task regression targets: $($qm9Selected.Count)" -ForegroundColor White
Write-Host "ToxCast multi-task classification targets: $($toxcastSelected.Count)" -ForegroundColor White
Write-Host "TOTAL MULTI-TASK SEARCHES TO PERFORM: $totalTests" -ForegroundColor Yellow
Write-Host ""

# Counter for progress tracking
$completedTests = 0
$failedTests = @()

# Function to run multi-task Optuna search
function Run-MultiTaskOptunaSearch {
    param(
        [string]$TaskType,
        [string]$DatasetName,
        [string[]]$Targets,
        [int]$TestNumber
    )
    
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "MULTI-TASK SEARCH ${TestNumber}/${totalTests}: $TaskType - $DatasetName" -ForegroundColor Green
    Write-Host "Targets ($($Targets.Count)): $($Targets -join ', ')" -ForegroundColor Green
    Write-Host "Global features: $UseGlobalFeatures" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    
    # Build command
    $cmd = "python optuna_search_main.py --task $TaskType --dataset_name $DatasetName --n_trials $NTrials --multitask"
    
    if ($TaskType -eq "regression") {
        $cmd += " --multitask_targets $($Targets -join ',')"
    } else {
        $cmd += " --multitask_assays $($Targets -join ',')"
    }
    
    # Apply subset settings - skip for ToxCast as it's already small
    if ($UseSubset -and $DatasetName -ne "TOXCAST") {
        $cmd += " --use_subset --subset_ratio $SubsetRatio"
        Write-Host "Note: Using subset for faster tuning (ToxCast skipped - dataset already small)" -ForegroundColor Yellow
    } elseif ($DatasetName -eq "TOXCAST") {
        Write-Host "Note: Skipping subset for ToxCast - dataset is already small" -ForegroundColor Yellow
    }
    
    if ($UseGlobalFeatures) {
        $cmd += " --use_global_features"
    }
    
    Write-Host "Executing: $cmd" -ForegroundColor Cyan
    
    try {
        Invoke-Expression $cmd
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: Multi-task $TaskType - $DatasetName" -ForegroundColor Green
        } else {
            Write-Host "FAILED: Multi-task $TaskType - $DatasetName (Exit code: $LASTEXITCODE)" -ForegroundColor Red
            $script:failedTests += "Multi-task $TaskType - $DatasetName"
        }
    } catch {
        Write-Host "FAILED: Multi-task $TaskType - $DatasetName (Error: $($_.Exception.Message))" -ForegroundColor Red
        $script:failedTests += "Multi-task $TaskType - $DatasetName"
    }
    
    $script:completedTests++
    
    # Progress update
    $progress = ($TestNumber / $totalTests) * 100
    $elapsed = (Get-Date) - $startTime
    if ($TestNumber -gt 0) {
        $avgTime = $elapsed.TotalSeconds / $TestNumber
        $remainingTime = $avgTime * ($totalTests - $TestNumber)
        $progressStr = $progress.ToString('F1')
        Write-Host "Progress: $TestNumber/$totalTests - $progressStr percent" -ForegroundColor Yellow
        Write-Host "Elapsed: $($elapsed.ToString('hh\:mm\:ss')), Estimated remaining: $([TimeSpan]::FromSeconds($remainingTime).ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
    }
    Write-Host ""
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RUNNING MULTI-TASK OPTUNA SEARCHES" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# QM8 multi-task regression
Run-MultiTaskOptunaSearch -TaskType "regression" -DatasetName "QM8" -Targets $qm8Selected -TestNumber 1

# QM9 multi-task regression
Run-MultiTaskOptunaSearch -TaskType "regression" -DatasetName "QM9" -Targets $qm9Selected -TestNumber 2

# ToxCast multi-task classification
Run-MultiTaskOptunaSearch -TaskType "classification" -DatasetName "TOXCAST" -Targets $toxcastSelected -TestNumber 3

$endTime = Get-Date
$totalTime = $endTime - $startTime

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "MULTI-TASK OPTUNA SEARCH COMPLETED!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Total execution time: $($totalTime.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "Completed searches: $($totalTests - $failedTests.Count)/$totalTests" -ForegroundColor White
if ($failedTests.Count -gt 0) {
    Write-Host "Failed searches: $($failedTests.Count)" -ForegroundColor Red
    Write-Host "Failed tests:" -ForegroundColor Red
    foreach ($failed in $failedTests) {
        Write-Host "  - $failed" -ForegroundColor Red
    }
}
Write-Host ""
Write-Host "Results saved in: experiments/optuna_search/" -ForegroundColor Yellow
Write-Host "Use python analyze_optuna_results.py to analyze results" -ForegroundColor White
