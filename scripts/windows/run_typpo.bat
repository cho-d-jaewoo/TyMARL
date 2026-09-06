@echo off
setlocal enabledelayedexpansion
call conda activate env_hetgrl
REM repo root
cd /d %~dp0..\..

REM Stagger startup so multiple parallel SC2 instances don't all bind ports
REM at the same instant.
ping -n 3 127.0.0.1 >nul

REM ============================================================
REM  USER CONFIG -- edit these as needed
REM ============================================================

REM Space-separated seeds. For paper-quality 5-seed averages:
REM   set SEEDS=1 2 3 4 5
set SEEDS=1

REM Resume mode:
REM   RESUME=0  -> always run, even if a previous run produced output
REM                (DEFAULT - safest while iterating).
REM   RESUME=1  -> skip experiments whose results\runs\<exp>\done.flag
REM                exists. Use this only if a long multi-day run got
REM                interrupted and you want to continue.
REM Note: done.flag is written by harl.train ONLY when training reaches
REM       num_env_steps. Crashed / Ctrl+C-killed runs do NOT produce one,
REM       so RESUME=1 will correctly re-run them from scratch.
set RESUME=0

REM ============================================================
REM  Experiments  --  TyPPO across all three races
REM ============================================================

echo ========================================================
echo  TyPPO on SMACv2 / protoss + terran + zerg
echo  n_rollout_threads=1 (set in harl\configs\algos_cfgs\typpo.yaml)
echo  SEEDS=%SEEDS%   RESUME=%RESUME%
echo  Started at %TIME%
echo ========================================================
echo.

for %%S in (%SEEDS%) do (
    call :run typpo smacv2 protoss_5_vs_5 %%S
    call :run typpo smacv2 terran_5_vs_5  %%S
    call :run typpo smacv2 zerg_5_vs_5    %%S
)

echo.
echo ========================================================
echo  TyPPO slot finished at %TIME%
echo ========================================================
goto :eof


:run
set ALGO=%~1
set ENV=%~2
set _SCEN=%~3
set SEED=%~4
set EXP_NAME=%ALGO%_%ENV%_%_SCEN%_seed%SEED%
set DONE_FLAG=results\runs\%EXP_NAME%\done.flag

if "%RESUME%"=="1" (
    if exist "%DONE_FLAG%" goto :skip
)

echo.
echo ----- %ALGO% on %ENV%/%_SCEN% seed=%SEED% started %TIME% -----

python -m harl.train --algo %ALGO% --env %ENV% --scenario %_SCEN% --algo-config harl\configs\algos_cfgs\%ALGO%.yaml --env-config harl\configs\envs_cfgs\%ENV%.yaml --seed %SEED% --exp-name %EXP_NAME% --device cuda --save-replay

set RC=!ERRORLEVEL!
if %RC% NEQ 0 goto :run_failed

echo TyPPO DONE %EXP_NAME% finished at %TIME%
exit /b 0

:skip
echo TyPPO SKIP %EXP_NAME% already completed (done.flag exists)
exit /b 0

:run_failed
echo TyPPO WARN %EXP_NAME% failed exit %RC% continuing
exit /b 0
