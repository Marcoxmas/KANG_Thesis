# Comprehensive Optuna hyperparameter search script for all datasets
# PowerShell version for Windows
# This script automatically runs Optuna searches for every dataset/target combination

param(
    [int]$NTrials = 20,
    [switch]$UseSubset = $false,
    [double]$SubsetRatio = 0.3,
    [switch]$UseGlobalFeatures = $false,
    [switch]$FirstTargetOnly = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "Usage: .\run_all_optuna_searches.ps1 [OPTIONS]" -ForegroundColor Green
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -NTrials NUM           Number of Optuna trials per search (default: 20)" -ForegroundColor White
    Write-Host "  -UseSubset             Use subset during hyperparameter search" -ForegroundColor White
    Write-Host "  -SubsetRatio RATIO     Ratio of dataset to use for subset (default: 0.3)" -ForegroundColor White
    Write-Host "  -UseGlobalFeatures     Use global molecular features" -ForegroundColor White
    Write-Host "  -FirstTargetOnly       Test only the first target from each dataset (for quick testing)" -ForegroundColor White
    Write-Host "  -Help                  Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_all_optuna_searches.ps1                              # Run with default settings" -ForegroundColor White
    Write-Host "  .\run_all_optuna_searches.ps1 -NTrials 50                  # Run with 50 trials per search" -ForegroundColor White
    Write-Host "  .\run_all_optuna_searches.ps1 -UseGlobalFeatures           # Use global features" -ForegroundColor White
    Write-Host "  .\run_all_optuna_searches.ps1 -FirstTargetOnly             # Quick test mode" -ForegroundColor White
    Write-Host "  .\run_all_optuna_searches.ps1 -UseSubset -SubsetRatio 0.2  # Use 20% subset during search" -ForegroundColor White
    exit 0
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "KANG THESIS - COMPREHENSIVE OPTUNA HYPERPARAMETER SEARCH" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting comprehensive Optuna search across all datasets..." -ForegroundColor Green
Write-Host "Trials per search: $NTrials" -ForegroundColor White
Write-Host "Use subset: $UseSubset" -ForegroundColor White
if ($UseSubset) {
    Write-Host "Subset ratio: $SubsetRatio" -ForegroundColor White
}
Write-Host "Use global features: $UseGlobalFeatures" -ForegroundColor White
if ($FirstTargetOnly) {
    Write-Host "QUICK TEST MODE: Testing only the first target from each dataset" -ForegroundColor Yellow
}
Write-Host ""

# Configuration
$RESULTS_DIR = "experiments/optuna_search"

# Create directories
New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null

# Get start time
$startTime = Get-Date

# Define all datasets and targets
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "DEFINING TEST CONFIGURATIONS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# QM8 targets (regression)
$qm8Targets = @("E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM")
if ($FirstTargetOnly) {
    $qm8Targets = @($qm8Targets[0])  # Just E1-CC2
}

# QM9 targets (regression)
$qm9Targets = @("mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv")
if ($FirstTargetOnly) {
    $qm9Targets = @($qm9Targets[0])  # Just mu
}

# ToxCast targets (classification)
$toxcastAssays = @(
    "TOX21_AhR_LUC_Agonist",
    "TOX21_Aromatase_Inhibition", 
    "TOX21_AutoFluor_HEK293_Cell_blue",
    "TOX21_p53_BLA_p3_ch1",
    "TOX21_p53_BLA_p4_ratio"
)
if ($FirstTargetOnly) {
    $toxcastAssays = @($toxcastAssays[0])  # Just TOX21_AhR_LUC_Agonist
}

$totalTests = $qm8Targets.Count + $qm9Targets.Count + 1 + $toxcastAssays.Count  # +1 for HIV

Write-Host "QM8 regression targets: $($qm8Targets.Count)" -ForegroundColor White
Write-Host "QM9 regression targets: $($qm9Targets.Count)" -ForegroundColor White
Write-Host "HIV classification: 1" -ForegroundColor White
Write-Host "ToxCast classification targets: $($toxcastAssays.Count)" -ForegroundColor White
Write-Host "TOTAL SEARCHES TO PERFORM: $totalTests" -ForegroundColor Yellow
Write-Host "Estimated time: $($totalTests * 30) minutes (assuming 30 min per search)" -ForegroundColor Yellow
Write-Host ""

# Counter for progress tracking
$completedTests = 0
$failedTests = @()

# Function to run individual Optuna search
function Run-OptunaSearch {
    param(
        [string]$TaskType,
        [string]$DatasetName,
        [string]$TargetColumn,
        [int]$TestNumber
    )
    
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "SEARCH ${TestNumber}/${totalTests}: $TaskType - $DatasetName" -ForegroundColor Green
    if ($TargetColumn) {
        Write-Host "Target: $TargetColumn" -ForegroundColor Green
    }
    Write-Host "Global features: $UseGlobalFeatures" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    
    # Build command
    $cmd = "python optuna_search_main.py --task $TaskType --dataset_name $DatasetName --n_trials $NTrials"
    
    if ($TargetColumn) {
        $cmd += " --target_column $TargetColumn"
    }
    
    if ($UseSubset) {
        $cmd += " --use_subset --subset_ratio $SubsetRatio"
    }
    
    if ($UseGlobalFeatures) {
        $cmd += " --use_global_features"
    }
    
    Write-Host "Executing: $cmd" -ForegroundColor Cyan
    
    try {
        Invoke-Expression $cmd
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: $TaskType - $DatasetName" -ForegroundColor Green
        } else {
            Write-Host "FAILED: $TaskType - $DatasetName (Exit code: $LASTEXITCODE)" -ForegroundColor Red
            $script:failedTests += "$TaskType - $DatasetName - $TargetColumn"
        }
    } catch {
        Write-Host "FAILED: $TaskType - $DatasetName (Error: $($_.Exception.Message))" -ForegroundColor Red
        $script:failedTests += "$TaskType - $DatasetName - $TargetColumn"
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
Write-Host "RUNNING OPTUNA SEARCHES" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$testNumber = 0

# QM8 regression
foreach ($target in $qm8Targets) {
    $testNumber++
    Run-OptunaSearch -TaskType "regression" -DatasetName "QM8" -TargetColumn $target -TestNumber $testNumber
}

# QM9 regression  
foreach ($target in $qm9Targets) {
    $testNumber++
    Run-OptunaSearch -TaskType "regression" -DatasetName "QM9" -TargetColumn $target -TestNumber $testNumber
}

# HIV classification
$testNumber++
Run-OptunaSearch -TaskType "classification" -DatasetName "HIV" -TargetColumn $null -TestNumber $testNumber

# ToxCast classification
foreach ($assay in $toxcastAssays) {
    $testNumber++
    Run-OptunaSearch -TaskType "classification" -DatasetName "TOXCAST" -TargetColumn $assay -TestNumber $testNumber
}

$endTime = Get-Date
$totalTime = $endTime - $startTime

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "OPTUNA SEARCH COMPLETED!" -ForegroundColor Green
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
