@echo off
REM Wipe ALL prior training results so we can restart fresh after the
REM InvalidMapData fix. This deletes:
REM   - results\runs\       (all training logs and checkpoints)
REM   - results\replays\    (all SC2 replay files)
REM   - results\figures\    (regenerable plots)
REM
REM SC2 temp maps from corrupted runs (in %TEMP%\StarCraft II\) are also
REM cleared so a stale TempLaunchMap.SC2Map can't poison the next launch.

setlocal
REM repo root
cd /d %~dp0..\..

echo.
echo This will permanently delete:
echo   - results\runs\        (training logs)
echo   - results\replays\     (SC2 replays)
echo   - results\figures\     (plots)
echo   - %%TEMP%%\StarCraft II\ (stale SC2 temp maps)
echo.
echo Press Ctrl+C now to abort, or any key to proceed.
pause >nul

if exist results\runs    rmdir /s /q results\runs
if exist results\replays rmdir /s /q results\replays
if exist results\figures rmdir /s /q results\figures

mkdir results\runs    2>nul
mkdir results\replays 2>nul
mkdir results\figures 2>nul

REM Clear SC2 temp maps; harmless if folder is in use (will skip locked files).
if exist "%TEMP%\StarCraft II" rmdir /s /q "%TEMP%\StarCraft II" 2>nul

REM Also clear any TyMARL per-process temp dirs from prior interrupted runs.
for /d %%D in ("%TEMP%\TyMARL_p*") do rmdir /s /q "%%D" 2>nul

echo.
echo Done. You can now relaunch training with run_all_parallel.bat
echo (or any individual run_slot_*_*.bat).
