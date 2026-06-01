@echo off
setlocal
set "REPO_ROOT=%~dp0.."
set "APP_ROOT=%~dp0"
set "NODE_ROOT=%REPO_ROOT%\.local-node\node-v24.16.0-win-x64"
set "NODE_EXE=%NODE_ROOT%\node.exe"
set "NEXT_CLI=%APP_ROOT%node_modules\next\dist\bin\next"
set "NPM_CLI=%NODE_ROOT%\node_modules\npm\bin\npm-cli.js"

if not exist "%NPM_CLI%" (
  echo Local npm CLI not found at "%NPM_CLI%".
  exit /b 1
)

if not exist "%NODE_EXE%" (
  echo Local node executable not found at "%NODE_EXE%".
  exit /b 1
)

set "PATH=%NODE_ROOT%;%PATH%"

if "%~1"=="" (
  "%NODE_EXE%" "%NEXT_CLI%" dev
) else if /I "%~1"=="run" (
  if /I "%~2"=="dev" (
    shift
    shift
    "%NODE_EXE%" "%NEXT_CLI%" dev %*
    exit /b %ERRORLEVEL%
  )
  if /I "%~2"=="build" (
    shift
    shift
    "%NODE_EXE%" "%NEXT_CLI%" build %*
    exit /b %ERRORLEVEL%
  )
  if /I "%~2"=="start" (
    shift
    shift
    "%NODE_EXE%" "%NEXT_CLI%" start %*
    exit /b %ERRORLEVEL%
  )
  "%NODE_EXE%" "%NPM_CLI%" %*
  exit /b %ERRORLEVEL%
) else (
  "%NODE_EXE%" "%NPM_CLI%" %*
)

exit /b %ERRORLEVEL%
