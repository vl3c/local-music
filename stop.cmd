@echo off
rem Stop the local-music server (and its engine workers) and release their VRAM.
rem
rem Normally you just close the start.cmd console window. Use this when there
rem is no window to close, e.g. the server was launched in the background.
rem
rem The server writes its PID to output\.server.pid; this stops that process tree,
rem whichever path (this folder or a junction to it) started it. As a fallback it
rem also stops python processes running server\app.py or server\workers\sa3_worker.py
rem from this folder. Other python programs are left alone.

setlocal
echo Looking for a running local-music server...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$here = '%~dp0'.TrimEnd('\'); $n = 0;" ^
  "$pidFile = Join-Path $here 'output\.server.pid';" ^
  "if (Test-Path $pidFile) {" ^
  "  $id = 0; [int]::TryParse((Get-Content $pidFile -Raw).Trim(), [ref]$id) | Out-Null;" ^
  "  $proc = Get-CimInstance Win32_Process -Filter \"ProcessId=$id\";" ^
  "  if ($proc -and $proc.CommandLine -match 'server.app\.py') {" ^
  "    Write-Host ('  stopping server PID {0} and its children' -f $id) -ForegroundColor Yellow;" ^
  "    taskkill /T /F /PID $id | Out-Null; $n++ };" ^
  "  Remove-Item $pidFile -ErrorAction SilentlyContinue };" ^
  "$roots = @($here); $t = (Get-Item -LiteralPath $here).Target;" ^
  "if ($t) { $roots += @($t | ForEach-Object { $_.TrimEnd('\') }) };" ^
  "Start-Sleep -Milliseconds 500;" ^
  "$p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {" ^
  "  $c = $_.CommandLine; $c -and ($c -match 'server.(app|workers.sa3_worker)\.py') -and" ^
  "  @($roots | Where-Object { $c.IndexOf($_, [StringComparison]::OrdinalIgnoreCase) -ge 0 }).Count };" ^
  "foreach ($x in $p) {" ^
  "  Write-Host ('  stopping PID {0}' -f $x.ProcessId) -ForegroundColor Yellow;" ^
  "  Stop-Process -Id $x.ProcessId -Force -ErrorAction SilentlyContinue; $n++ };" ^
  "if (-not $n) { Write-Host '  Nothing running.' -ForegroundColor DarkGray; exit 0 };" ^
  "Start-Sleep -Seconds 2;" ^
  "Write-Host '  Stopped.' -ForegroundColor Green"

if exist "%SystemRoot%\System32\nvidia-smi.exe" (
  echo.
  echo VRAM now in use:
  "%SystemRoot%\System32\nvidia-smi.exe" --query-gpu=memory.used,memory.total --format=csv,noheader
)

endlocal
