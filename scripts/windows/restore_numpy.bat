@echo off
REM repo root
cd /d %~dp0..\..
REM Restore numpy to a version compatible with torch on Windows.
REM Run this if you see "torch import" or DLL loading errors.
call conda activate env_hetgrl
pip install --upgrade --force-reinstall numpy==1.26.4
echo Done. Try running tests again.
