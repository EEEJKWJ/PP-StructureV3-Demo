@echo off
setlocal
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
cd /d "%~dp0PP-StructureV3-Demo"
"C:\anaconda3\envs\PP-StructureV3_Demo\python.exe" server.py >> "%~dp0PP-StructureV3-Demo.server.log" 2>&1
