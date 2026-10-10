@echo off
rem Start the local-music server in its own window.
rem Close that window to stop the server and release the GPU.
rem
rem   start.cmd                 the server opens in its own window and this returns at once;
rem                             the page opens in your browser once it's ready
rem   start.cmd --no-browser    the same without opening the page (what agents use)
rem   start.cmd --lazy          load nothing until the first request (the page comes up in
rem                             seconds; the first track then waits ~40 s for the engine)
rem
rem The server only ever runs in that window, so it can't be left running out of sight: it
rem refuses to start anywhere else (a hidden background shell, python serverpp.py).
setlocal
cd /d "%~dp0"
rem Captured before the option loop: it shifts the arguments (shift /1 leaves %0 alone,
rem but keeping these in variables makes that impossible to get wrong later).
set "SELF=%~f0"
set "HERE_DIR=%~dp0"

rem Windows' own timeout. Started from Git Bash, plain "timeout" can resolve to the GNU
rem coreutils one, which doesn't understand /t.
set "WAIT=%SystemRoot%\System32\timeout.exe"

if not exist "%HERE_DIR%.venv\Scripts\python.exe" (
  echo.
  echo No install found. Run install.ps1 first.
  echo.
  pause
  exit /b 1
)

set "BROWSER=1"
set "INWIN="
set "EXTRA="
:args
if "%~1"=="" goto argsdone
if /i "%~1"=="--no-browser" (set "BROWSER=" & shift /1 & goto args)
if /i "%~1"=="--lazy"       (set "EXTRA=--lazy" & shift /1 & goto args)
if /i "%~1"=="--in-window"  (set "INWIN=1" & shift /1 & goto args)
echo Unknown option "%~1". Use:  start.cmd [--no-browser] [--lazy]
exit /b 2
:argsdone

rem Relaunch in a new, visible console and return immediately, so the server is always on
rem screen and closing its window shuts it down (Windows ends every process attached to a
rem console when the console closes). LOCAL_MUSIC_WINDOW is how the server knows it is in
rem that window: the new console inherits it from here; nothing else sets it.
if defined INWIN goto run
set "PASS=%EXTRA%"
if not defined BROWSER set "PASS=--no-browser %EXTRA%"
set "LOCAL_MUSIC_WINDOW=1"
start "local-music server - close this window to stop it" cmd /c ""%SELF%" --in-window %PASS%"
exit /b 0

:run
(
  title local-music server - close this window to stop it
  echo.
  echo   local-music server on http://127.0.0.1:8192/
  echo   Close this window to stop it and free the GPU.
  echo.
)

rem Wait for the server to answer before opening the browser: the model takes
rem a little while to load.
if defined BROWSER start "" /min cmd /c "for /l %%i in (1,1,150) do (curl.exe -s -o nul --max-time 2 http://127.0.0.1:8192/api/health && (start "" http://127.0.0.1:8192/ & exit /b) || "%WAIT%" /t 2 >nul)"

rem The server binds to 127.0.0.1 only (hard-coded in server\app.py).
"%HERE_DIR%.venv\Scripts\python.exe" "%HERE_DIR%server\app.py" --port 8192 %EXTRA%

(
  echo.
  echo   The server has stopped ^(exit code %ERRORLEVEL%^). This window closes in 60 seconds;
  echo   press a key to close it now.
  "%WAIT%" /t 60 >nul
)

endlocal
