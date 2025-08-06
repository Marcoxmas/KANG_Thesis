# Comprehensive multi-seed testing script for all datasets and columns
# PowerShell version for Windows
# This script automatically runs multi-seed tests for every dataset/column combination
# and generates comprehensive summaries

param(
    [int]$NumSeeds = 5,
    [int]$Epochs = 50,
    [int]$Patience = 25,
    [switch]$FirstTargetOnly = $false
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "KANG THESIS - COMPREHENSIVE MULTI-SEED TESTING SUITE" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting comprehensive multi-seed testing across all datasets..." -ForegroundColor Green
if ($FirstTargetOnly) {
    Write-Host "QUICK TEST MODE: Testing only the first target from each dataset" -ForegroundColor Yellow
}
Write-Host ""

# Configuration
$RESULTS_DIR = "experiments/multi_seed_results"
$SUMMARY_DIR = "experiments/multi_seed_summaries"

# Create directories
New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null
New-Item -ItemType Directory -Path $SUMMARY_DIR -Force | Out-Null

# Counter for tracking progress
$totalTests = 0
$completedTests = 0

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "PHASE 1: COUNTING TOTAL TESTS TO PERFORM" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# QM8 targets (regression)
$qm8Targets = @("E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM")
if ($FirstTargetOnly) {
    $qm8Targets = @($qm8Targets[0])  # Just E1-CC2
}
$totalTests += $qm8Targets.Count
Write-Host "QM8 regression targets: $($qm8Targets.Count)" -ForegroundColor White

# QM9 targets (regression)
$qm9Targets = @("mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv")
if ($FirstTargetOnly) {
    $qm9Targets = @($qm9Targets[0])  # Just mu
}
$totalTests += $qm9Targets.Count
Write-Host "QM9 regression targets: $($qm9Targets.Count)" -ForegroundColor White

# HIV (classification)
$totalTests += 1
Write-Host "HIV classification: 1" -ForegroundColor White

# ToxCast targets (classification) - Focus on specific 5 assays
Write-Host "Using predefined ToxCast assays for thesis..." -ForegroundColor Green

# Define the 5 specific ToxCast assays for the thesis
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

Write-Host "Selected ToxCast assays for thesis:" -ForegroundColor White
for ($i = 0; $i -lt $toxcastAssays.Count; $i++) {
    Write-Host "$($i+1). $($toxcastAssays[$i])" -ForegroundColor White
}

$totalTests += $toxcastAssays.Count
Write-Host "ToxCast classification targets: $($toxcastAssays.Count)" -ForegroundColor White

if ($FirstTargetOnly) {
    Write-Host ""
    Write-Host "FIRST TARGET ONLY MODE - Selected targets:" -ForegroundColor Yellow
    Write-Host "  QM8: $($qm8Targets[0])" -ForegroundColor White
    Write-Host "  QM9: $($qm9Targets[0])" -ForegroundColor White
    Write-Host "  HIV: HIV (classification)" -ForegroundColor White
    Write-Host "  ToxCast: $($toxcastAssays[0])" -ForegroundColor White
}

Write-Host ""
Write-Host "TOTAL TESTS TO PERFORM: $totalTests" -ForegroundColor Yellow
Write-Host "Estimated time: $($totalTests * 15) minutes (assuming 15 min per test)" -ForegroundColor Yellow
Write-Host ""

# Function to run a single test and handle errors
function Run-SingleTest {
    param(
        [string]$TaskType,
        [string]$DatasetName,
        [string]$TargetColumn,
        [string]$TestName
    )
    
    $script:completedTests++
    Write-Host "[$script:completedTests/$totalTests] Running: $TestName" -ForegroundColor Cyan
    Write-Host "  Task: $TaskType | Dataset: $DatasetName | Target: $TargetColumn" -ForegroundColor Gray
    
    # Create a unique log file for this test
    $logFile = "$RESULTS_DIR/${TaskType}_${DatasetName}"
    if ($TargetColumn -ne "") {
        $logFile += "_${TargetColumn}"
    }
    $logFile += "_${NumSeeds}seeds.log"
    
    # Build the command
    $cmd = "python", "multi_seed_test.py", "--task", $TaskType, "--dataset_name", $DatasetName, "--num_seeds", $NumSeeds, "--epochs", $Epochs, "--patience", $Patience
    if ($TargetColumn -ne "") {
        $cmd += "--target_column", $TargetColumn
    }
    
    # Run the test with error handling
    try {
        $output = & $cmd[0] $cmd[1..($cmd.Length-1)] 2>&1
        $output | Out-File -FilePath $logFile -Encoding UTF8
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  SUCCESS" -ForegroundColor Green
        } else {
            Write-Host "  FAILED - Check $logFile for details" -ForegroundColor Red
            "FAILED: $TestName" | Add-Content -Path "$RESULTS_DIR/failed_tests.txt"
        }
    } catch {
        Write-Host "  FAILED - $($_.Exception.Message)" -ForegroundColor Red
        "FAILED: $TestName - $($_.Exception.Message)" | Add-Content -Path "$RESULTS_DIR/failed_tests.txt"
    }
    
    Write-Host "  Progress: $script:completedTests/$totalTests completed" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "PHASE 2: RUNNING MULTI-SEED TESTS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Initialize failed tests file
"# Failed Tests Log" | Out-File -FilePath "$RESULTS_DIR/failed_tests.txt" -Encoding UTF8
"# Generated on $(Get-Date)" | Add-Content -Path "$RESULTS_DIR/failed_tests.txt"
"" | Add-Content -Path "$RESULTS_DIR/failed_tests.txt"

$startTime = Get-Date

Write-Host ""
Write-Host "--- QM8 REGRESSION TESTS ---" -ForegroundColor Yellow
foreach ($target in $qm8Targets) {
    Run-SingleTest -TaskType "regression" -DatasetName "QM8" -TargetColumn $target -TestName "QM8-$target"
}

Write-Host ""
Write-Host "--- QM9 REGRESSION TESTS ---" -ForegroundColor Yellow
foreach ($target in $qm9Targets) {
    Run-SingleTest -TaskType "regression" -DatasetName "QM9" -TargetColumn $target -TestName "QM9-$target"
}

Write-Host ""
Write-Host "--- HIV CLASSIFICATION TEST ---" -ForegroundColor Yellow
Run-SingleTest -TaskType "classification" -DatasetName "HIV" -TargetColumn "" -TestName "HIV"

Write-Host ""
Write-Host "--- TOXCAST CLASSIFICATION TESTS ---" -ForegroundColor Yellow
foreach ($assay in $toxcastAssays) {
    Run-SingleTest -TaskType "classification" -DatasetName $assay -TargetColumn "" -TestName "ToxCast-$assay"
}

$endTime = Get-Date
$totalTime = ($endTime - $startTime).TotalSeconds

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "PHASE 3: GENERATING COMPREHENSIVE SUMMARIES" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "Running comprehensive summary generation..." -ForegroundColor Green
try {
    python generate_comprehensive_summary.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Summary generation completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Summary generation failed!" -ForegroundColor Red
    }
} catch {
    Write-Host "Error running summary generation: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "COMPREHENSIVE MULTI-SEED TESTING COMPLETE!" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "EXECUTION SUMMARY:" -ForegroundColor Yellow
Write-Host "  Total tests attempted: $totalTests" -ForegroundColor White
Write-Host "  Total tests completed: $completedTests" -ForegroundColor White
$hours = [math]::Floor($totalTime / 3600)
$minutes = [math]::Floor(($totalTime % 3600) / 60)
$seconds = [math]::Floor($totalTime % 60)
Write-Host "  Total execution time: ${hours}h ${minutes}m ${seconds}s" -ForegroundColor White
Write-Host ""
Write-Host "RESULTS LOCATIONS:" -ForegroundColor Yellow
Write-Host "  Individual results: $RESULTS_DIR/" -ForegroundColor White
Write-Host "  Comprehensive summaries: $SUMMARY_DIR/" -ForegroundColor White
Write-Host "  Test logs: $RESULTS_DIR/*.log" -ForegroundColor White
Write-Host ""

# Check for failed tests
if (Test-Path "$RESULTS_DIR/failed_tests.txt") {
    $failedContent = Get-Content "$RESULTS_DIR/failed_tests.txt" | Where-Object { $_.StartsWith("FAILED:") }
    $failedCount = $failedContent.Count
    if ($failedCount -gt 0) {
        Write-Host "WARNING: $failedCount tests failed. Check $RESULTS_DIR/failed_tests.txt for details." -ForegroundColor Red
    }
}

Write-Host "To view the main summary:" -ForegroundColor Yellow
Write-Host "  Get-Content $SUMMARY_DIR/comprehensive_summary.txt" -ForegroundColor White
Write-Host ""
Write-Host "To view performance plots:" -ForegroundColor Yellow
Write-Host "  Start-Process $SUMMARY_DIR/performance_overview.png" -ForegroundColor White
Write-Host "  Start-Process $SUMMARY_DIR/global_features_impact.png" -ForegroundColor White
Write-Host ""
Write-Host "To view LaTeX table:" -ForegroundColor Yellow
Write-Host "  Get-Content $SUMMARY_DIR/latex_summary_table.txt" -ForegroundColor White
Write-Host ""

Write-Host "Script execution completed successfully!" -ForegroundColor Green
