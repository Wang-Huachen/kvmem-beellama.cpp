@echo off
REM kvmem-kvarn-tools.bat
REM
REM Stage 3 batch 8 gate: the KVMem CLI / server must be ABLE TO TURN KVarN ON.
REM
REM Why this exists: until batch 8 both tools built llama_context_params from
REM scratch and never assigned cparams.kvarn, and their -ctk parser only accepted
REM f16/q8_0/q5_0/q4_0/f32. So params.kvarn stayed DISABLED, the context silently
REM ran the plain row cache, and "KVMem + KVarN" was impossible from a real command
REM line -- every KVarN result up to then came from the smoke test, which sets
REM cparams.kvarn directly through the public API.
REM
REM What each case is for:
REM   A  --kvarn off                       -> control: no KVarN anywhere
REM   B  --kvarn kvarn_k4v4_g128           -> the arena is really constructed
REM   C  -ctk kvarn_k4v4_g128              -> the ordinary flag accepts KVarN names
REM   D  MTP model + --kvarn + draft-mtp   -> target arena AND the MTP follower
REM
REM Usage:
REM   scripts\nightly\kvmem-kvarn-tools.bat <kvarn-model.gguf> [mtp-model.gguf]
REM   set KVMEM_TOOLS_MODEL=<model.gguf>
REM   set KVMEM_TOOLS_MTP_MODEL=<mtp-model.gguf>
REM
REM Exit codes: 0 = PASS, 1 = FAIL, 77 = SKIP (no model path).
setlocal enabledelayedexpansion
cd /d "%~dp0\..\.."
set "ROOT=%CD%"

set "CMAKE_EXE=%ProgramFiles%\CMake\bin\cmake.exe"
if not exist "%CMAKE_EXE%" (
    for /f "delims=" %%i in ('where cmake 2^>nul') do set "CMAKE_EXE=%%i"
)
if defined BUILD_DIR (
    set "BUILD=%BUILD_DIR%"
) else (
    if exist "%ROOT%\build-win" (set "BUILD=%ROOT%\build-win") else (set "BUILD=%ROOT%\build")
)
set "TYPE=Release"
set "BIN=%BUILD%\bin\%TYPE%"
set "CLI=%BIN%\llama-kvmem-cli.exe"

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%KVMEM_TOOLS_MODEL%"
if "%MODEL%"=="" (
    echo SKIP: no model path.
    echo       usage: scripts\nightly\kvmem-kvarn-tools.bat ^<kvarn-model.gguf^> [mtp-model.gguf]
    endlocal
    exit /b 77
)
if not exist "%MODEL%" (
    echo SKIP: model not found: %MODEL%
    endlocal
    exit /b 77
)
set "MTP=%~2"
if "%MTP%"=="" set "MTP=%KVMEM_TOOLS_MTP_MODEL%"
if not "%MTP%"=="" if not exist "%MTP%" (
    echo SKIP: model not found: %MTP%
    endlocal
    exit /b 77
)

echo === [1/2] build the tools ===
REM No -j: the Visual Studio generator already parallelises through MSBuild.
"%CMAKE_EXE%" --build "%BUILD%" --config %TYPE% --target llama-kvmem-cli llama-kvmem-server
if %errorlevel% neq 0 (
    echo RESULT: FAIL - build failed.
    endlocal
    exit /b 1
)
if not exist "%CLI%" (
    echo RESULT: FAIL - expected binary not found: %CLI%
    endlocal
    exit /b 1
)

echo.
echo === [2/2] run the KVarN tool matrix ===
set "FAILED=0"
set "OUT=%TEMP%\kvmem-tools-out.txt"
set "COMMON=--kvmem --kvmem-method recency --kvmem-block-tokens 128 -n 8 -ngl 99 --temp 0 --no-prompt"

REM A: control. A plain ggml row cache, so nothing may report a KVarN cache.
set "OPTS=--kv-dtype q8_0 %COMMON%"
call :case "A --kv-dtype q8_0 (control)" "kvarn=0"
call :absent "A --kv-dtype q8_0 (control)" "enabling structured KVarN cache type"

REM B/C: the two spellings a beellama user already knows. There is no --kvarn
REM switch anywhere: KVarN is a cache TYPE, exactly like kvarn6 in beellama.cpp.
set "OPTS=-ctk kvarn4 %COMMON%"
call :case "B -ctk kvarn4" "KVMEM kvarN target cache:" "enabling structured KVarN cache type" "kvarn=1"
call :absent "B -ctk kvarn4" "MTP follower is not ported"

set "OPTS=--cache-type-k kvarn6 --cache-type-v kvarn6 %COMMON%"
call :case "C --cache-type-k kvarn6" "KVMEM kvarN target cache:" "kvarn=1"

REM D: the real deployment shape -- target arena plus the MTP follower.
if not "%MTP%"=="" (
    set "MODEL=%MTP%"
    set "OPTS=-ctk kvarn4 --spec-type draft-mtp --spec-draft-n-max 2 %COMMON%"
    call :case "D MTP model + KVarN + draft-mtp" "kvarn=1" "mtp_pool cells="
) else (
    echo.
    echo --- D MTP model: not given, skipped ---
)

echo.
echo --- summary: failures=%FAILED% ---
if "%FAILED%"=="0" (
    echo RESULT: PASS
    endlocal
    exit /b 0
)
echo RESULT: FAIL
endlocal
exit /b 1

REM ---- run one case from %MODEL% / %OPTS%, require exit 0 and each marker ----
REM %1 label, %2..%4 markers. Each marker is matched with findstr /C: against the
REM captured stdout+stderr of this run.
:case
set "LABEL=%~1"
echo.
echo --- %LABEL% ---
"%CLI%" -m "%MODEL%" %OPTS% "Hello my name is" > "%OUT%" 2>&1
set "RC=!errorlevel!"
set "OK=1"
if not "!RC!"=="0" (
    echo FAIL: exit code !RC!
    set "OK=0"
)
call :check "%~2"
call :check "%~3"
call :check "%~4"
if "!OK!"=="0" (
    type "%OUT%"
    set /a FAILED+=1
) else (
    echo ok: exit 0, all markers present
)
goto :eof

REM ---- require a marker in %OUT% ----
:check
if "%~1"=="" goto :eof
findstr /C:"%~1" "%OUT%" >nul
if !errorlevel! neq 0 (
    echo FAIL: missing marker: %~1
    set "OK=0"
) else (
    echo   ok: %~1
)
goto :eof

REM ---- require a marker to be ABSENT in %OUT% (the run just captured) ----
:absent
echo.
echo --- %~1: marker must be absent ---
findstr /C:"%~2" "%OUT%" >nul
if !errorlevel! equ 0 (
    echo FAIL: unexpected marker present: %~2
    set /a FAILED+=1
) else (
    echo   ok: absent %~2
)
goto :eof
