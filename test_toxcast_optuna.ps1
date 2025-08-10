# Test ToxCast Optuna searches only
# PowerShell version for Windows

param(
    [int]$NTrials = 20,
    [switch]$UseSubset = $false,
    [double]$SubsetRatio = 0.3,
    [switch]$UseGlobalFeatures = $false,
    [switch]$FirstTargetOnly = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "Usage: .\test_toxcast_optuna.ps1 [OPTIONS]" -ForegroundColor Green
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -NTrials NUM           Number of Optuna trials per search (default: 20)" -ForegroundColor White
    Write-Host "  -UseSubset             Use subset during hyperparameter search" -ForegroundColor White
    Write-Host "  -SubsetRatio RATIO     Ratio of dataset to use for subset (default: 0.3)" -ForegroundColor White
    Write-Host "  -UseGlobalFeatures     Use global molecular features" -ForegroundColor White
    Write-Host "  -FirstTargetOnly       Test only the first target (TOX21_AhR_LUC_Agonist)" -ForegroundColor White
    Write-Host "  -Help                  Show this help message" -ForegroundColor White
    exit 0
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "TESTING TOXCAST OPTUNA SEARCHES" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Trials per search: $NTrials" -ForegroundColor White
Write-Host "Use subset: $UseSubset" -ForegroundColor White
if ($UseSubset) {
    Write-Host "Subset ratio: $SubsetRatio" -ForegroundColor White
}
Write-Host "Use global features: $UseGlobalFeatures" -ForegroundColor White
if ($FirstTargetOnly) {
    Write-Host "TESTING MODE: Testing only TOX21_AhR_LUC_Agonist" -ForegroundColor Yellow
}
Write-Host ""

# Configuration
$RESULTS_DIR = "experiments/optuna_search"
New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null

# Get start time
$startTime = Get-Date

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

$totalTests = $toxcastAssays.Count
$failedTests = @()

Write-Host "ToxCast classification targets to test: $($toxcastAssays.Count)" -ForegroundColor White
Write-Host "TOTAL SEARCHES TO PERFORM: $totalTests" -ForegroundColor Yellow
Write-Host ""

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
    Write-Host "Target: $TargetColumn" -ForegroundColor Green
    Write-Host "Global features: $UseGlobalFeatures" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    
    # Build command
    $cmd = "python optuna_search_main.py --task $TaskType --dataset_name $DatasetName --n_trials $NTrials"
    $cmd += " --target_column $TargetColumn"
    
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
            Write-Host "SUCCESS: $TaskType - $DatasetName - $TargetColumn" -ForegroundColor Green
        } else {
            Write-Host "FAILED: $TaskType - $DatasetName - $TargetColumn (Exit code: $LASTEXITCODE)" -ForegroundColor Red
            $script:failedTests += "$TaskType - $DatasetName - $TargetColumn"
        }
    } catch {
        Write-Host "FAILED: $TaskType - $DatasetName - $TargetColumn (Error: $($_.Exception.Message))" -ForegroundColor Red
        $script:failedTests += "$TaskType - $DatasetName - $TargetColumn"
    }
    
    Write-Host ""
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RUNNING TOXCAST OPTUNA SEARCHES" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$testNumber = 0

# ToxCast classification
foreach ($assay in $toxcastAssays) {
    $testNumber++
    Run-OptunaSearch -TaskType "classification" -DatasetName "TOXCAST" -TargetColumn $assay -TestNumber $testNumber
}

$endTime = Get-Date
$totalTime = $endTime - $startTime

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "TOXCAST OPTUNA SEARCH COMPLETED!" -ForegroundColor Green
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
