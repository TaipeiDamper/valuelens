$ErrorActionPreference = "Stop"

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    throw "找不到 pyinstaller，請先執行 pip install pyinstaller"
}

pyinstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --name ValueLens `
  main.py

Write-Host "Build completed: dist\\ValueLens.exe"
