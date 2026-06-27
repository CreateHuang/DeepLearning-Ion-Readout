$ErrorActionPreference = "Continue"
Set-Location "F:\QIP-article-template\DL_For_300Qubit_Readout"
$py = "D:\anaconda3\envs\py312\python.exe"
$seeds = @(1,2,3)
$foldEpochs = "150,400,600,800,1000"
$logRoot = "F:\QIP-article-template\DL_For_300Qubit_Readout\outputs\seed_training_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$env:PYTHONUNBUFFERED = "1"
foreach ($seed in $seeds) {
    $outDir = "F:\QIP-article-template\DL_For_300Qubit_Readout\outputs\train_unet_multiion_seed$seed"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $log = Join-Path $logRoot "seed${seed}.log"
    "[$(Get-Date -Format o)] START seed=$seed outDir=$outDir foldEpochs=$foldEpochs" | Tee-Object -FilePath $log -Append
    & $py -u train_main.py --model dwnet --seed $seed --load_mode full --out_dir $outDir --fold_epochs $foldEpochs 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        "[$(Get-Date -Format o)] FAILED seed=$seed exit=$LASTEXITCODE" | Tee-Object -FilePath $log -Append
        exit $LASTEXITCODE
    }
    "[$(Get-Date -Format o)] DONE seed=$seed" | Tee-Object -FilePath $log -Append
}
"[$(Get-Date -Format o)] ALL_DONE" | Tee-Object -FilePath (Join-Path $logRoot "all_done.log") -Append

