@echo off
setlocal enabledelayedexpansion
call conda activate env_hetgrl
REM repo root
cd /d %~dp0..\..

ping -n 3 127.0.0.1 >nul

REM ============================================================
REM  MPE: 5 algorithms x 2 scenarios, single terminal sequential.
REM  Following HARL (Zhong et al., JMLR 2024) MPE evaluation protocol:
REM    PPO-family algorithms -> DISCRETE action mode
REM    SAC-family algorithms -> CONTINUOUS action mode
REM
REM  Order:
REM    PPO-family (disc):  typpo, happo, mappo
REM    SAC-family (cont):  tysac, hasac
REM  Scenarios per algo:  simple_spread_v3 -> simple_speaker_listener_v4
REM ============================================================

set SEEDS=1
set RESUME=0

echo ========================================================
echo  MPE: 5 algos x 2 scenarios, sequential
echo  (PPO -> discrete, SAC -> continuous)
echo  num_env_steps=2M (algo configs: <algo>_mpe.yaml)
echo  SEEDS=%SEEDS%   RESUME=%RESUME%
echo  Started at %TIME%
echo ========================================================

for %%S in (%SEEDS%) do (
    REM PPO-family: discrete MPE
    call :run typpo spread           simple_spread_v3           mpe_simple_spread_disc     %%S
    call :run typpo speaker_listener simple_speaker_listener_v4 mpe_speaker_listener_disc  %%S
    call :run happo spread           simple_spread_v3           mpe_simple_spread_disc     %%S
    call :run happo speaker_listener simple_speaker_listener_v4 mpe_speaker_listener_disc  %%S
    call :run mappo spread           simple_spread_v3           mpe_simple_spread_disc     %%S
    call :run mappo speaker_listener simple_speaker_listener_v4 mpe_speaker_listener_disc  %%S
    REM SAC-family: continuous MPE
    call :run tysac spread           simple_spread_v3           mpe_simple_spread_cont     %%S
    call :run tysac speaker_listener simple_speaker_listener_v4 mpe_speaker_listener_cont  %%S
    call :run hasac spread           simple_spread_v3           mpe_simple_spread_cont     %%S
    call :run hasac speaker_listener simple_speaker_listener_v4 mpe_speaker_listener_cont  %%S
)

echo ========================================================
echo  MPE all-algos slot finished at %TIME%
echo ========================================================
goto :eof


:run
set ALGO=%~1
set _NICK=%~2
set _SCEN=%~3
set _ENVCFG=%~4
set SEED=%~5
set EXP_NAME=%ALGO%_mpe_%_NICK%_seed%SEED%
set DONE_FLAG=results\runs\%EXP_NAME%\done.flag

if "%RESUME%"=="1" (
    if exist "%DONE_FLAG%" goto :skip
)

echo.
echo ----- %ALGO% on mpe/%_NICK% (%_SCEN% via %_ENVCFG%) seed=%SEED% started %TIME% -----

python -m harl.train --algo %ALGO% --env mpe --scenario %_SCEN% --algo-config harl\configs\algos_cfgs\%ALGO%_mpe.yaml --env-config harl\configs\envs_cfgs\%_ENVCFG%.yaml --seed %SEED% --exp-name %EXP_NAME% --device cuda

set RC=!ERRORLEVEL!
if %RC% NEQ 0 goto :run_failed

echo MPE DONE %EXP_NAME% finished at %TIME%
exit /b 0

:skip
echo MPE SKIP %EXP_NAME% already completed
exit /b 0

:run_failed
echo MPE WARN %EXP_NAME% failed exit %RC% continuing
exit /b 0
