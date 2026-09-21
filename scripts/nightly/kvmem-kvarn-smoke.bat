@echo off
REM kvmem-kvarn-smoke.bat
REM
REM Stage 3 batch 3 gate: KVMem with its inner cache swapped onto the KVarN
REM record arena. Builds and runs tests\kvmem-kvarn-smoke.cpp, which loads a
REM real GGUF, prefills through KVMem's slot pool, decodes a few tokens off the
REM arena and prints PASS/FAIL.
REM
REM This target links llama ONLY (never llama-common), so it can be built in the
REM normal build tree even while llama-common.lib is locked by another process.
REM
REM Usage:
REM   scripts\nightly\kvmem-kvarn-smoke.bat <model.gguf>
REM   set KVMEM_SMOKE_MODEL=<model.gguf>  ^&^&  scripts\nightly\kvmem-kvarn-smoke.bat
REM
REM Exit codes: 0 = PASS, 1 = FAIL, 77 = SKIP (no model path given).
setlocal
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
set "TARGET=kvmem-kvarn-smoke"
set "EXE=%BUILD%\bin\%TYPE%\%TARGET%.exe"
if not exist "%EXE%" set "EXE=%BUILD%\bin\%TARGET%.exe"
if not exist "%EXE%" if exist "%ROOT%\build-win\bin\%TARGET%.exe" set "EXE=%ROOT%\build-win\bin\%TARGET%.exe"
if not exist "%EXE%" if exist "%ROOT%\build\bin\Release\%TARGET%.exe" set "EXE=%ROOT%\build\bin\Release\%TARGET%.exe"

if "%~1"=="" if not defined KVMEM_SMOKE_MODEL (
    echo SKIP: no model path.
    echo       usage: scripts\nightly\kvmem-kvarn-smoke.bat ^<model.gguf^>
    echo       or set KVMEM_SMOKE_MODEL first.
    endlocal
    exit /b 77
)

echo === [1/2] build %TARGET% in "%BUILD%" ===
REM No -j: the Visual Studio generator parallelizes through MSBuild already.
"%CMAKE_EXE%" --build "%BUILD%" --config %TYPE% --target %TARGET%
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
echo === [2/2] run %EXE% ===
pushd "%ROOT%"
if "%~1"=="" (
    "%EXE%" "%KVMEM_SMOKE_MODEL%"
) else (
    "%EXE%" "%~1"
)
set "RC=%errorlevel%"
popd

echo.
echo --- exit code: %RC% ---
if "%RC%"=="0" (
    echo RESULT: PASS
    endlocal
    exit /b 0
)
if "%RC%"=="77" (
    echo RESULT: SKIP
    endlocal
    exit /b 77
)
echo RESULT: FAIL
endlocal
exit /b 1