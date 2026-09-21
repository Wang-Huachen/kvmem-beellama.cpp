@echo off
REM kvmem-kvarn-hybrid.bat
REM
REM Stage 3 batch 6 gate: KVMem on a HYBRID architecture (attention layers + GDN
REM recurrent layers) with the KVarN record arena.
REM
REM Why this is a separate script from kvmem-kvarn-evict.bat: that one is the dense
REM regression matrix. The hybrid path is a different code path -- the memory object
REM is llama_memory_kvmem_hybrid, which composes a llama_memory_kvmem for the
REM attention half and moves it into llama_memory_hybrid. Until batch 6 that wrapper
REM borrowed the base class's stock llama_kv_cache, so the base's init_kv_batch()
REM never reached KVMem and hybrid + KVarN had to be refused outright. These cases
REM are what proves the composition actually landed.
REM
REM What each case is for:
REM   A  row cache (KVarN off), hybrid                       -> the pre-batch-6 behaviour
REM   B  KVarN record arena, hybrid, identity pool           -> the new path
REM   C  KVarN record arena, hybrid, prefill pressure        -> eviction + the virtual
REM                                                             slot numbering are live
REM
REM A hybrid model reports PASS only if the log shows all three of
REM   KVarN cache constructed / KVMem took over / KVMem owns a record arena,
REM so a silent fallback to the stock memory is a FAIL, not a skip.
REM
REM Usage:
REM   scripts\nightly\kvmem-kvarn-hybrid.bat <hybrid-model.gguf>
REM   set KVMEM_HYBRID_MODEL=<model.gguf>  ^&^&  scripts\nightly\kvmem-kvarn-hybrid.bat
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
set "TARGET=kvmem-kvarn-smoke"
set "EXE=%BUILD%\bin\%TYPE%\%TARGET%.exe"
if not exist "%EXE%" set "EXE=%BUILD%\bin\%TARGET%.exe"
if not exist "%EXE%" if exist "%ROOT%\build-win\bin\%TARGET%.exe" set "EXE=%ROOT%\build-win\bin\%TARGET%.exe"
if not exist "%EXE%" if exist "%ROOT%\build\bin\Release\%TARGET%.exe" set "EXE=%ROOT%\build\bin\Release\%TARGET%.exe"

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%KVMEM_HYBRID_MODEL%"
if "%MODEL%"=="" (
    echo SKIP: no model path.
    echo       usage: scripts\nightly\kvmem-kvarn-hybrid.bat ^<hybrid-model.gguf^>
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
echo === [2/2] run the hybrid matrix ===
pushd "%BUILD%\bin\%TYPE%"

set "FAILED=0"

call :case "A hybrid, row cache (KVarN off)"        -ctk q8_0 --kv-dtype q8_0 --block-tokens 128 --ctx 512
call :case "B hybrid + KVarN arena"                 -ctk kvarn4 --block-tokens 128 --ctx 512
call :case "C hybrid + KVarN arena + pressure"      -ctk kvarn4 --block-tokens 128 --ctx 1024 --budget 384 --prefill 512

popd

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

REM ---- run one configuration, expect exit 0 ----
REM %* after shift is every argument except the label, so long option lists work.
REM A hybrid run has no SKIP state any more: either KVMem took the record arena or
REM the smoke test reports FAIL itself.
:case
set "LABEL=%~1"
shift
echo.
echo --- %LABEL% ---
"%EXE%" "%MODEL%" %* > "%TEMP%\kvmem-hybrid-out.txt" 2>&1
set "RC=%errorlevel%"
findstr /C:"RESULT:" "%TEMP%\kvmem-hybrid-out.txt"
if "%RC%"=="0" goto :eof
echo FAIL: exit code %RC%
type "%TEMP%\kvmem-hybrid-out.txt"
set /a FAILED+=1
goto :eof
