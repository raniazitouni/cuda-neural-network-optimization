@echo off
REM =============================================================================
REM run_optimized.bat
REM Windows — Run our optimized CUDA strategy on all 3 datasets
REM Requires: CUDA Toolkit + nvcc in PATH
REM =============================================================================

set BINARY=nn_optimized.exe
set DATA_DIR=data
set RESULTS_DIR=results
set RESULTS_FILE=%RESULTS_DIR%\optimized.csv

REM ---------------------------------------------------------------------------
REM 1. Build if needed
REM ---------------------------------------------------------------------------
if not exist %BINARY% (
    echo [BUILD] Binary not found - compiling...
    nvcc -O2 -Xcompiler -fopenmp -o nn_optimized src/optimized_cuda.cu -lm
    if errorlevel 1 (
        echo [ERROR] Compilation failed. Make sure CUDA Toolkit and nvcc are installed.
        pause
        exit /b 1
    )
    echo [OK] Built: %BINARY%
)

REM ---------------------------------------------------------------------------
REM 2. Check data
REM ---------------------------------------------------------------------------
for %%S in (small medium large) do (
    if not exist %DATA_DIR%\synthetic_convex_%%S.csv (
        echo [ERROR] Missing %DATA_DIR%\synthetic_convex_%%S.csv
        echo Run:    python playground\generate_synthetic_data.py
        pause
        exit /b 1
    )
)

REM ---------------------------------------------------------------------------
REM 3. Run
REM ---------------------------------------------------------------------------
if not exist %RESULTS_DIR% mkdir %RESULTS_DIR%
echo dataset,time_seconds> %RESULTS_FILE%

echo.
echo === Optimized Strategy: Tiling + Transpose + Kernel Fusion ===
echo.

for %%S in (small medium large) do (
    echo [%%S] Running on %DATA_DIR%\synthetic_convex_%%S.csv ...
    %BINARY% %DATA_DIR%\synthetic_convex_%%S.csv
    echo.
)

echo Results saved to %RESULTS_FILE%
echo Run run_compare.bat to compare against the reference baseline.
pause