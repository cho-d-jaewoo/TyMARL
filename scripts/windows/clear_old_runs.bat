@echo off
REM repo root
cd /d %~dp0..\..
REM Clear all results so the new runs start fresh.
REM Run this BEFORE run_slot_A.bat / run_slot_B.bat if you want to discard
REM old logs from the pre-refactor codebase.
echo Clearing results\runs ...
if exist results\runs rmdir /s /q results\runs
mkdir results\runs
echo Done. Old run logs removed.
