@echo off
REM kvarn-move-selftest.bat
REM
REM Builds and runs the model-free KVarN "whole crate move" self test:
REM   tests\kvarn-move-selftest.cpp  ->  build-win\bin\kvarn-move-selftest.exe
REM
REM The test seals a KVarN record group with ggml_kvarn_store, copies its bytes
REM to a second slot without re-quantizing, reads both back with
REM ggml_kvarn_materialize and asserts the results are bit identical (plus a
REM negative control that must differ).
REM
REM Exit code: 0 when the exe reports PASS, non-zero otherwise.
setlocal
cd /d "%~dp0\..\.."

set "ROOT=%CD%"

REM ---- CUDA 环境（与 scripts\build-cuda.bat 一致，仅用于编译期找 nvcc） ----
if defined CU13 (
    if exist "%CU13%\bin\nvcc.exe" (
        set "PATH=%CU13%\bin;%PATH%"
        set "LD_LIBRARY_PATH=%CU13%\lib;%LD_LIBRARY_PATH%"
        set "CUDA_HOME=%CU13%"
        set "CMAKE_CUDA_COMPILER=%CU13%\bin\nvcc.exe"
    )
)
if not defined CMAKE_CUDA_COMPILER (
    set "CMAKE_CUDA_COMPILER=nvcc"
)

REM ---- CMake 可执行文件（与 scripts\build-cuda.bat 相同的查找顺序） ----
if defined CMAKE (
    set "CMAKE_EXE=%CMAKE%"
) else (
    set "CMAKE_EXE=%ROOT%\.venv\Scripts\cmake.exe"
    if not exist "%CMAKE_EXE%" (
        for /f "delims=" %%i in ('where cmake 2^>nul') do set "CMAKE_EXE=%%i"
    )
)

REM ---- 构建目录与配置 ----
if defined BUILD_DIR (
    set "BUILD=%BUILD_DIR%"
) else (
    if exist "%ROOT%\build-win" (set "BUILD=%ROOT%\build-win") else (set "BUILD=%ROOT%\build")
)
if defined CMAKE_BUILD_TYPE (
    set "TYPE=%CMAKE_BUILD_TYPE%"
) else (
    set "TYPE=Release"
)
if defined NPROC (
    set "JOBS=%NPROC%"
) else (
    set "JOBS=%NUMBER_OF_PROCESSORS%"
)

set "TARGET=kvarn-move-selftest"
set "EXE=%BUILD%\bin\%TYPE%\%TARGET%.exe"
if not exist "%EXE%" set "EXE=%BUILD%\bin\%TARGET%.exe"
if not exist "%EXE%" if exist "%ROOT%\build-win\bin\%TARGET%.exe" set "EXE=%ROOT%\build-win\bin\%TARGET%.exe"
if not exist "%EXE%" if exist "%ROOT%\build\bin\Release\%TARGET%.exe" set "EXE=%ROOT%\build\bin\Release\%TARGET%.exe"

echo === [1/3] configure (only needed once, harmless to repeat) ===
"%CMAKE_EXE%" -S "%ROOT%" -B "%BUILD%" ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE="%TYPE%" ^
  -DCMAKE_CUDA_COMPILER="%CMAKE_CUDA_COMPILER%" ^
  -DCMAKE_CUDA_ARCHITECTURES="120a-real" ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DKVMEM_BUILD_LLAMA=ON ^
  -DLLAMA_KVMEM=ON ^
  -DLLAMA_KVMEM_ROOT="%ROOT%" ^
  -DCMAKE_CXX_FLAGS="/utf-8" ^
  -DLLAMA_BUILD_TESTS=OFF

if %errorlevel% neq 0 if not exist "%EXE%" (
    echo RESULT: FAIL - CMake configure failed.
    endlocal
    exit /b 1
)

echo.
echo === [2/3] build target %TARGET% ===
REM No -j here: the Visual Studio generator already parallelizes through MSBuild,
REM and passing -j alongside the VS generator fails the build.
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
echo === [3/3] run %EXE% ===
REM The exe sits next to ggml.dll / ggml-cuda.dll in build-win\bin, and the
REM process directory is searched first on Windows, so no PATH change is needed.
pushd "%BUILD%\bin\%TYPE%"
"%EXE%"
set "RC=%errorlevel%"
popd

echo.
echo --- exit code: %RC% ---
if "%RC%"=="0" (
    echo RESULT: PASS
    endlocal
    exit /b 0
)

echo RESULT: FAIL
endlocal
exit /b 1
