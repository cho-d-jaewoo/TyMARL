@echo off
REM Quick health-check: verify that the fixed smacv2_env.py is in place
REM and that StarCraft2Env can be constructed without kwarg errors.
call conda activate env_hetgrl
REM repo root
cd /d %~dp0..\..

echo === [1/3] Checking smacv2_env.py for the fix marker ===
findstr /C:"safe_kwargs" harl\envs\smacv2\smacv2_env.py >nul
if %ERRORLEVEL% NEQ 0 (
    echo FAIL: smacv2_env.py does NOT contain the fix.
    echo You are running the OLD code. Re-extract TyMARL_fixed.zip.
    exit /b 1
)
echo OK: smacv2_env.py has the fix.

echo.
echo === [2/3] Checking torch import ===
python -c "import torch; print(f'OK: torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"
if %ERRORLEVEL% NEQ 0 (
    echo FAIL: torch import failed. See restore_numpy.bat or reinstall torch.
    exit /b 1
)

echo.
echo === [3/3] Smoke-running typpo on smacv2 for 1 short iter ===
python -m harl.train --algo typpo --env smacv2 --scenario protoss_5_vs_5 --algo-config harl\configs\algos_cfgs\typpo.yaml --env-config harl\configs\envs_cfgs\smacv2.yaml --seed 1 --exp-name __healthcheck__ --device cpu
if %ERRORLEVEL% NEQ 0 (
    echo FAIL: training crashed. Inspect the traceback above.
    exit /b 1
)
echo.
echo OK: all checks passed.
