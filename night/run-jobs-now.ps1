# 夜間の処理（night/jobs.json）のうち、指定したものだけを今すぐ順に実行する（日中に手で動かすとき用。runner.mjs の時間帯の制限・完了の記録は使わない）。
# 使い方: powershell -File night/run-jobs-now.ps1 <id> [<id> ...]
# 完了の印は付けないが、対局のスクリプトは保存済みの区切りを飛ばすので、夜の runner が同じ処理を始めてもすぐ終わる。
param([Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Ids)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$jobs = Get-Content night\jobs.json -Raw -Encoding UTF8 | ConvertFrom-Json
$log = "D:\shogi_data\night\manual_$(Get-Date -Format yyyy-MM-dd).log"
function Log($s) { Add-Content -Path $log -Value "$(Get-Date -Format 'yyyy/MM/dd H:mm:ss') $s" -Encoding UTF8 }
foreach ($id in $Ids) {
  $job = $jobs | Where-Object id -eq $id
  if (-not $job) { Log "処理が見つかりません: $id"; exit 1 }
  Log "処理開始: $id（$($job.name)）"
  $args2 = $job.cmd[1..($job.cmd.Count - 1)]
  & $job.cmd[0] @args2 2>&1 | ForEach-Object { Add-Content -Path $log -Value $_ -Encoding UTF8 }
  if ($LASTEXITCODE -ne 0) { Log "処理が失敗（終了コード $LASTEXITCODE）: $id。ここで止める"; exit 1 }
  Log "処理完了: $id"
}
Log '終了'
