@echo off
REM =============================================================================
REM run_compare.bat
REM Windows — Build and compare reference vs optimized strategies
REM =============================================================================

set REF_BIN=nn_reference.exe
set OPT_BIN=nn_optimized.exe
set DATA_DIR=data
set RESULTS_DIR=results
set COMPARE_FILE=%RESULTS_DIR%\comparison.csv

REM ---------------------------------------------------------------------------
REM 1. Build reference if needed
REM ---------------------------------------------------------------------------
if not exist %REF_BIN% (
    echo [BUILD] Building reference baseline...
    nvcc -O2 -Xcompiler /openmp -o %REF_BIN% reference\nn_cuda.cu
    if errorlevel 1 (
        echo [ERROR] Failed to build reference. Check reference\nn_cuda.cu exists.
        pause
        exit /b 1
    )
    echo [OK] Built: %REF_BIN%
)

REM ---------------------------------------------------------------------------
REM 2. Build optimized if needed
REM ---------------------------------------------------------------------------
if not exist %OPT_BIN% (
    echo [BUILD] Building optimized alternative...
    nvcc -O2 -Xcompiler /openmp -o %OPT_BIN% src\tiling_optimized_transpose_fusion.cu
    if errorlevel 1 (
        echo [ERROR] Failed to build optimized. Check src\tiling_optimized_transpose_fusion.cu exists.
        pause
        exit /b 1
    )
    echo [OK] Built: %OPT_BIN%
)

REM ---------------------------------------------------------------------------
REM 3. Check data
REM ---------------------------------------------------------------------------
for %%S in (small medium large) do (
    if not exist %DATA_DIR%\synthetic_convex_%%S.csv (
        echo [ERROR] Missing %DATA_DIR%\synthetic_convex_%%S.csv
        echo Run:    python playground\generate_synthetic_data.py
        pause
        exit /b 1
    )
)

if not exist %RESULTS_DIR% mkdir %RESULTS_DIR%
echo dataset,reference_time_s,optimized_time_s > %COMPARE_FILE%

REM ---------------------------------------------------------------------------
REM 4. Run both on each dataset
REM ---------------------------------------------------------------------------
echo.
echo ================================================================
echo  CUDA Neural Network — Strategy Comparison
echo  Reference:  naive mat_mult (global memory only)
echo  Optimized:  tiling + transpose + fused forward pass
echo ================================================================
echo.

for %%S in (small medium large) do (
    echo ---- Dataset: %%S ----
    echo.
    echo [REFERENCE]
    %REF_BIN% %DATA_DIR%\synthetic_convex_%%S.csv
    echo.
    echo [OPTIMIZED]
    %OPT_BIN% %DATA_DIR%\synthetic_convex_%%S.csv
    echo.
)

echo Full results saved to: %COMPARE_FILE%
pause