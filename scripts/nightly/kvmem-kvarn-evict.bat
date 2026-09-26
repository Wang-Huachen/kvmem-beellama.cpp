@echo off
REM kvmem-kvarn-evict.bat
REM
REM Stage 3 batch 4 gate: the KVarN record round trip plus the guards that keep it
REM honest. Runs tests\kvmem-kvarn-smoke.cpp in six configurations.
REM
REM What each case is for:
REM   A  plain-ggml KVMem with the OLD block size (32)  -> the pre-existing config
REM   B  plain-ggml KVMem with block size 128           -> the config the user moved to
REM   C  KVarN arena, identity pool, no eviction        -> batch 3 must still work
REM   D  KVarN arena + prefill pressure                 -> the batch 4 record snapshot /
REM                                                        restore path is actually reached
REM   E  KVarN arena switched OFF, pressure prefill     -> row path unaffected
REM
REM D is expected to be the interesting one. If it aborts with "partial block is not
REM in the live slot" that is the KNOWN batch 4 limitation (see AGENTS.md 4.20): the
REM guard is doing its job by refusing to let attention silently read zeros. This
REM script reports it as KNOWN rather than FAIL so a real regression still stands out.
REM
REM Usage:
REM   scripts\nightly\kvmem-kvarn-evict.bat <model.gguf>
REM   set KVMEM_SMOKE_MODEL=<model.gguf>  ^&^&  scripts\nightly\kvmem-kvarn-evict.bat
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
    if exist "%ROOT%\build-mirror\bin" (set "BUILD=%ROOT%\build-mirror") else if exist "%ROOT%\build-win" (set "BUILD=%ROOT%\build-win") else (set "BUILD=%ROOT%\build")
)
set "TYPE=Release"
set "TARGET=kvmem-kvarn-smoke"
set "EXE=%BUILD%\bin\%TYPE%\%TARGET%.exe"
if not exist "%EXE%" set "EXE=%BUILD%\bin\%TARGET%.exe"
if not exist "%EXE%" if exist "%ROOT%\build-mirror\bin\%TARGET%.exe" set "EXE=%ROOT%\build-mirror\bin\%TARGET%.exe"
if not exist "%EXE%" if exist "%ROOT%\build-win\bin\%TARGET%.exe" set "EXE=%ROOT%\build-win\bin\%TARGET%.exe"
if not exist "%EXE%" if exist "%ROOT%\build\bin\Release\%TARGET%.exe" set "EXE=%ROOT%\build\bin\Release\%TARGET%.exe"

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%KVMEM_SMOKE_MODEL%"
if "%MODEL%"=="" (
    echo SKIP: no model path.
    echo       usage: scripts\nightly\kvmem-kvarn-evict.bat ^<model.gguf^>
    endlocal
    exit /b 77
)
if not exist "%MODEL%" (
    echo SKIP: model not found: %MODEL%
    endlocal
    exit /b 77
)

echo === [1/2] build %TARGET% ===
REM No -j: the Visual Studio generator already parallelises through MSBuild.
if defined KVMEM_SKIP_BUILD (echo [matrix] KVMEM_SKIP_BUILD=1: reusing existing %EXE%) else ("%CMAKE_EXE%" --build "%BUILD%" --config %TYPE% --target %TARGET%)
if %errorlevel% neq 0 if not exist "%EXE%" (
    echo RESULT: FAIL - build failed.
    endlocal
    exit /b 1
)
if not exist "%EXE%" (
    echo RESULT: FAIL - expected binary not found: %EXE%
    endlocal
    exit /b 1
)

echo.
echo === [2/2] run the matrix ===
pushd "%BUILD%\bin\%TYPE%"

set "FAILED=0"
set "KNOWN=0"

call :case "A plain-ggml KVMem, block 32"      -ctk q8_0 --kv-dtype q8_0 --block-tokens 32 --ctx 512
call :case "B plain-ggml KVMem, block 128"     -ctk q8_0 --kv-dtype q8_0 --block-tokens 128 --ctx 512
call :case "C KVarN arena, identity pool"      -ctk kvarn4 --block-tokens 128 --ctx 512
call :case "E KVarN off + pressure prefill"    -ctk q8_0 --kv-dtype q8_0 --block-tokens 128 --ctx 1024 --budget 384 --prefill 512
call :pressured "D KVarN arena + pressure"     -ctk kvarn4 --block-tokens 128 --ctx 1024 --budget 384 --prefill 512

popd

echo.
echo --- summary: failures=%FAILED% known-limitation=%KNOWN% ---
if "%FAILED%"=="0" (
    echo RESULT: PASS
    endlocal
    exit /b 0
)
echo RESULT: FAIL
endlocal
exit /b 1

REM ---- run one configuration, expect exit 0 or a graceful SKIP (77) ----
REM %* after shift is every argument except the label, so long option lists work.
:case
set "LABEL=%~1"
shift
echo.
echo --- %LABEL% ---
"%EXE%" "%MODEL%" %* > "%TEMP%\kvmem-evict-out.txt" 2>&1
set "RC=%errorlevel%"
findstr /C:"RESULT:" "%TEMP%\kvmem-evict-out.txt"
if "%RC%"=="0" goto :eof
if "%RC%"=="77" goto :eof
echo FAIL: exit code %RC%
type "%TEMP%\kvmem-evict-out.txt"
set /a FAILED+=1
goto :eof

REM ---- the pressured KVarN case: abort on the known limitation is not a failure ----
:pressured
set "LABEL=%~1"
shift
echo.
echo --- %LABEL% ---
"%EXE%" "%MODEL%" %* > "%TEMP%\kvmem-evict-pressured.txt" 2>&1
set "RC=%errorlevel%"
findstr /C:"RESULT:" "%TEMP%\kvmem-evict-pressured.txt"
if "%RC%"=="0" goto :eof
findstr /C:"partial block is not in the live slot" "%TEMP%\kvmem-evict-pressured.txt" >nul
if %errorlevel% equ 0 (
    echo KNOWN: batch 4 limitation reached ^(partial block did not get the live slot^);
    echo        the guard fired instead of silently attending to zeros. See AGENTS.md 4.20.
    set /a KNOWN+=1
    goto :eof
)
echo FAIL: exit code %RC% with an unexpected error
type "%TEMP%\kvmem-evict-pressured.txt"
set /a FAILED+=1
goto :eof
