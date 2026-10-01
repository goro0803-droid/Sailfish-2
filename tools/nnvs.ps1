# 新しいネットワーク（baseline フォルダの .nn）を、基準のネットワークと順番に対局させる
# 使い方: powershell -File tools/nnvs.ps1 <exe> <games> <ms> <par> <基準のネットワーク> <新しいネットワーク（カンマ区切り）>
param([string]$exe, [int]$games, [int]$ms, [int]$par, [string]$base, [string]$nets)
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($n in ($nets -split ',')) {
  "=== $n vs $base"
  node --no-warnings tools/match.mjs "$exe|NetworkDelay=0,Threads=1,EvalFile=$n" "$exe|NetworkDelay=0,Threads=1,EvalFile=$base" $games $ms $par 2>&1 | Select-Object -Last 5 | Select-Object -First 4
}
