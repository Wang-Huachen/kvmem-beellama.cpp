@echo off
REM kvmem-kvarn-retrieval.bat
REM
REM Stage 3 retrieval gate: does KVMem's retrieval scorer actually have data to
REM work with -- and does it say so when it does not?
REM
REM Why this exists: retrieval scores a block by dotting the captured query-sum
REM against the block's mean pre-RoPE K. Both halves are written ONLY from the graph
REM capture hooks (llm_graph_context::kvmem_capture_q / _k), and those hooks are
REM wired by exactly two model files: llama.cpp/src/models/qwen3.cpp and
REM llama.cpp/src/models/qwen35.cpp. On any other architecture the ledger stays
REM empty, every block scores 0, and the selector hands back a recency window while
REM the caller believes it got a retrieval -- silently.
REM
REM That silence is what these cases check:
REM   A  hooks model (Qwen3.5), KVarN off       -> retrieval must produce real scores
REM   B  hooks model, KVarN record arena        -> the same, through the arena
REM   C  hooks model, KVarN, prefill pressure   -> the same, with eviction live
REM   D  hookless model (Hunyuan), KVarN on     -> must REPORT "N/A" plus a
REM                                                KVMEM_RETRIEVAL_DEGRADED line,
REM                                                never a fake PASS
REM
REM Case B is the regression that matters: KVarN once looked like the cause of dead
REM retrieval, when the real cause was the test model having no capture hooks at all.
REM
REM Usage:
REM   scripts\nightly\kvmem-kvarn-retrieval.bat <hooks-model.gguf> [hookless-model.gguf]
REM   set KVMEM_RETRIEVAL_MODEL=<hooks-model.gguf>
REM   set KVMEM_DENSE_MODEL=<hookless-model.gguf>
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

set "HOOKS=%~1"
if "%HOOKS%"=="" set "HOOKS=%KVMEM_RETRIEVAL_MODEL%"
if "%HOOKS%"=="" (
    echo SKIP: no capture-hook model path.
    echo       usage: scripts\nightly\kvmem-kvarn-retrieval.bat ^<hooks-model.gguf^> [hookless-model.gguf]
    endlocal
    exit /b 77
)
if not exist "%HOOKS%" (
    echo SKIP: model not found: %HOOKS%
    endlocal
    exit /b 77
)
set "DENSE=%~2"
if "%DENSE%"=="" set "DENSE=%KVMEM_DENSE_MODEL%"
if not "%DENSE%"=="" if not exist "%DENSE%" (
    echo SKIP: model not found: %DENSE%
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
echo === [2/2] run the retrieval matrix ===
pushd "%BUILD%\bin\%TYPE%"

set "FAILED=0"

REM A capture-hook arch must produce real scores, on both cache formats.
set "MODEL=%HOOKS%"
call :case "A hooks model, row cache (KVarN off)" "retrieval   : ok" "-ctk q8_0 --kv-dtype q8_0 --block-tokens 128 --ctx 512 --method retrieval --do-retrieval"
call :case "B hooks model + KVarN arena"          "retrieval   : ok" "-ctk kvarn4 --block-tokens 128 --ctx 512 --method retrieval --do-retrieval"
call :case "C hooks model + KVarN + pressure"     "retrieval   : ok" "-ctk kvarn4 --ctx 1024 --budget 384 --prefill 512 --method retrieval --do-retrieval"

REM A hookless arch cannot retrieve on ANY cache format. The gate is that it says
REM so, in both the per-run summary and the once-per-context degraded report.
if not "%DENSE%"=="" (
    set "MODEL=%DENSE%"
    call :case "D hookless model + KVarN (expect N/A)" "retrieval   : N/A" "-ctk kvarn4 --block-tokens 128 --ctx 512 --method retrieval --do-retrieval"
    call :marker "D hookless model + KVarN (degraded report)" "KVMEM_RETRIEVAL_DEGRADED reason=no-capture-hooks"
) else (
    echo.
    echo --- D hookless model: not given, skipped ---
)

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

REM ---- run one configuration with %MODEL%, expect exit 0 plus a marker in stdout ----
REM The option list travels as a single quoted argument, so there is no shift / %*
REM juggling; the model comes from %MODEL%, which each case sets before calling.
:case
set "LABEL=%~1"
set "EXPECT=%~2"
set "OPTS=%~3"
echo.
echo --- %LABEL% ---
"%EXE%" "%MODEL%" %OPTS% > "%TEMP%\kvmem-retr-out.txt" 2>&1
set "RC=%errorlevel%"
findstr /C:"RESULT:" "%TEMP%\kvmem-retr-out.txt"
if not "%RC%"=="0" (
    echo FAIL: exit code %RC%
    type "%TEMP%\kvmem-retr-out.txt"
    set /a FAILED+=1
    goto :eof
)
findstr /C:"%EXPECT%" "%TEMP%\kvmem-retr-out.txt" >nul
if %errorlevel% neq 0 if not exist "%EXE%" (
    echo FAIL: expected marker not found: %EXPECT%
    type "%TEMP%\kvmem-retr-out.txt"
    set /a FAILED+=1
    goto :eof
)
echo ok: %EXPECT%
goto :eof

REM ---- expect a second marker in the output of the run just made ----
:marker
echo.
echo --- %~1 ---
findstr /C:"%~2" "%TEMP%\kvmem-retr-out.txt" >nul
if %errorlevel% neq 0 if not exist "%EXE%" (
    echo FAIL: expected marker not found: %~2
    type "%TEMP%\kvmem-retr-out.txt"
    set /a FAILED+=1
    goto :eof
)
echo ok: %~2
goto :eof
