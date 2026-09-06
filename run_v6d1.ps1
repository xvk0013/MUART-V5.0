param(
    [ValidateSet('Check','Generate','Audit','Smoke','Train')][string]$Mode='Check',
    [string]$Python='python',
    [string]$Matlab='matlab',
    [string]$DataRoot='',
    [string]$DataGenDir='',
    [string]$ModelDir='',
    [string]$OutputRoot='',
    [string]$ReferenceRun='',
    [string]$OutputDir='',
    [int]$NumWorkers=4
)
$ErrorActionPreference='Stop'

if(-not $ModelDir) { $ModelDir=$PSScriptRoot }
if(-not $DataGenDir) { $DataGenDir=Join-Path $ModelDir 'data_gen_v2' }
if(-not $DataRoot) { $DataRoot=$env:MUART_DATA_ROOT }
if(-not $OutputRoot) { $OutputRoot=$env:MUART_OUTPUT_ROOT }
if(-not $DataRoot) {
    throw 'DataRoot is required. Pass -DataRoot or set MUART_DATA_ROOT.'
}

$referenceData=Join-Path $DataRoot 'data_cargo_sl_sir15/SS2_bp12/dataset'
$d1Data=Join-Path $DataRoot 'data_cargo_sl_sir15_d1_recording_uniform/SS2_bp12/dataset'
$templateRoot=Join-Path $DataRoot 'data_sl_templates_bp12'
$splitRoot=Join-Path $DataRoot 'data_sl_split_bp12'
if(-not $ReferenceRun -and $OutputRoot) {
    $ReferenceRun=Join-Path $OutputRoot 'checkpoints_v6a2_shared_evidence_s42'
}

Write-Host "D1 DataRoot  : $DataRoot"
Write-Host "D1 DataGenDir: $DataGenDir"
Write-Host "D1 ModelDir  : $ModelDir"
Write-Host "D1 OutputRoot: $(if($OutputRoot){$OutputRoot}else{'<not set>'})"
Write-Host "D1 A2 reference: $(if($ReferenceRun){$ReferenceRun}else{'<not set>'})"

$dataFiles=@(
    'build_source_sampling_pool.m','sample_source_from_pool.m','gen_sl_dataset.m',
    'd1_recording_uniform_config.m','main_cargo_sl_d1.m',
    'validate_d1_recording_uniform.m','test_d1_recording_uniform.m',
    'audit_d1_dataset.py','test_audit_d1_dataset.py'
)
$modelFiles=@('config_v6d1.py','d1_verdict.py','train_v6d1.py',
    'test_d1_verdict.py','test_v6d1.py')
foreach($name in $dataFiles) {
    $requiredFile=Join-Path $DataGenDir $name
    if(-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Missing D1 data file: $requiredFile"
    }
}
foreach($name in $modelFiles) {
    $requiredFile=Join-Path $ModelDir $name
    if(-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Missing D1 model file: $requiredFile"
    }
}
foreach($requiredRoot in @($referenceData,$templateRoot,$splitRoot)) {
    if(-not (Test-Path -LiteralPath $requiredRoot -PathType Container)) {
        throw "Missing D1 input directory: $requiredRoot"
    }
}

$env:MUART_DATA_ROOT=$DataRoot
$matlabDir=$DataGenDir.Replace("'","''")

if($Mode -eq 'Check') {
    Push-Location $DataGenDir
    try {
        & $Python -m py_compile 'audit_d1_dataset.py' 'test_audit_d1_dataset.py'
        if($LASTEXITCODE -ne 0) { throw 'D1 audit compile failed' }
        & $Python -m unittest -v test_audit_d1_dataset
        if($LASTEXITCODE -ne 0) { throw 'D1 audit tests failed' }
    } finally { Pop-Location }
    & $Matlab -batch "cd('$matlabDir'); validate_d1_recording_uniform; test_d1_recording_uniform"
    if($LASTEXITCODE -ne 0) { throw 'D1 MATLAB checks failed' }
    Push-Location $ModelDir
    try {
        & $Python -m py_compile 'config_v6d1.py' 'd1_verdict.py' 'train_v6d1.py' `
            'test_d1_verdict.py' 'test_v6d1.py'
        if($LASTEXITCODE -ne 0) { throw 'D1 training compile failed' }
        & $Python -m unittest -v test_d1_verdict test_v6d1 test_v6a2 test_v6a_conditional_set
        if($LASTEXITCODE -ne 0) { throw 'D1 training tests failed' }
    } finally { Pop-Location }
    Write-Host 'CHECK PASS: synthetic/unit checks only; no formal data generated.'
    exit 0
}

if($Mode -eq 'Generate') {
    & $Matlab -batch "cd('$matlabDir'); main_cargo_sl_d1"
    if($LASTEXITCODE -ne 0) { throw 'D1 generation failed' }
    Write-Host "GENERATE PASS: $d1Data"
    exit 0
}

if($Mode -eq 'Audit') {
    Push-Location $DataGenDir
    try {
        & $Python 'audit_d1_dataset.py' `
            --reference-data-dir $referenceData `
            --d1-data-dir $d1Data `
            --template-root $templateRoot `
            --split-root $splitRoot
        if($LASTEXITCODE -ne 0) { throw 'D1 dataset audit failed' }
    } finally { Pop-Location }
    Write-Host "AUDIT PASS: $(Join-Path $d1Data 'd1_audit.md')"
    exit 0
}

$auditPath=Join-Path $d1Data 'd1_audit.json'
if(-not (Test-Path -LiteralPath $auditPath -PathType Leaf)) {
    throw "Run -Mode Audit first: $auditPath"
}
$audit=Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
if($audit.status -ne 'PASS') { throw "D1 audit status is $($audit.status)" }
if($Mode -eq 'Train' -and (-not $ReferenceRun -or
        -not (Test-Path -LiteralPath $ReferenceRun -PathType Container))) {
    throw "Missing formal V6-A2 reference result: $ReferenceRun. Pass -ReferenceRun or -OutputRoot."
}

Push-Location $ModelDir
try {
    if(-not $OutputDir) {
        if(-not $OutputRoot) {
            throw 'OutputRoot is required for Smoke/Train when OutputDir is not supplied.'
        }
        if($Mode -eq 'Smoke') {
            $stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
            $OutputDir=Join-Path $OutputRoot "V6D1_smoke_s42_$stamp"
        } else {
            $stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
            $OutputDir=Join-Path $OutputRoot "checkpoints_v6d1_recording_uniform_s42_$stamp"
        }
    }
    Write-Host "D1 selected output: $OutputDir"
    $arguments=@('train_v6d1.py',
        '--data-dir',$d1Data,
        '--reference-data-dir',$referenceData,
        '--output-dir',$OutputDir,
        '--seed','42','--num-workers',"$NumWorkers")
    if($Mode -eq 'Smoke') {
        $arguments+='--smoke-test'
    } else {
        $arguments+=@('--reference-dir',$ReferenceRun)
    }
    & $Python @arguments
    if($LASTEXITCODE -ne 0) { throw "D1 $Mode failed (exit $LASTEXITCODE)" }
    Write-Host "$Mode PASS: $OutputDir"
} finally { Pop-Location }
