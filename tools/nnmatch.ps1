# ニューラルネットワーク（baseline フォルダの .nn）を使うエンジンと、手作りの評価関数のエンジンを順番に対局させる
# 使い方: powershell -File tools/nnmatch.ps1 <exe> <games> <ms> <par> <ネットワークのファイル名（カンマ区切り）> [基準のFeat=1]
param([string]$exe, [int]$games, [int]$ms, [int]$par, [string]$nets, [int]$base = 1)
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($n in ($nets -split ',')) {
  "=== $n vs 手作り(Feat=$base)"
  node --no-warnings tools/match.mjs "$exe|NetworkDelay=0,Threads=1,Feat=33,EvalFile=$n" "$exe|NetworkDelay=0,Threads=1,Feat=$base" $games $ms $par 2>&1 | Select-Object -Last 5 | Select-Object -First 4
}
