# 同じ実行ファイルで Feat（探索の改良のビット）を変えて、順番に自動対局する
# 使い方: powershell -File tools/ab.ps1 <exe> <games> <ms> <par> <基準のFeat> <比べるFeat（カンマ区切り）>
param([string]$exe, [int]$games, [int]$ms, [int]$par, [int]$base, [string]$tests)
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($t in ($tests -split ',')) {
  "=== Feat=$t vs Feat=$base"
  node --no-warnings tools/match.mjs "$exe|NetworkDelay=0,Feat=$t,Threads=1" "$exe|NetworkDelay=0,Feat=$base,Threads=1" $games $ms $par 2>&1 | Select-Object -Last 5 | Select-Object -First 4
}
