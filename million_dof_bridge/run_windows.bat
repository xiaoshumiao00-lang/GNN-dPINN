@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "VENV_DIR=.venv-windows"
set "PYEXE="

if defined PYTHON set "PYEXE=%PYTHON%"
if not defined PYEXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PYEXE=python"
    )
)
if not defined PYEXE (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PYEXE=py -3"
    )
)
if not defined PYEXE goto no_python

echo [INFO] Native Windows Python:
%PYEXE% -c "import sys; print(sys.executable); print(sys.version)"
%PYEXE% -c "import torch, numpy, matplotlib" >nul 2>&1
if errorlevel 1 goto setup_env
goto deps_ready

:setup_env
echo [INFO] Creating native Windows environment: %VENV_DIR%
if not exist "%VENV_DIR%\Scripts\python.exe" (
    %PYEXE% -m venv --system-site-packages "%VENV_DIR%"
    if errorlevel 1 goto venv_fail
)
set "PYEXE=%VENV_DIR%\Scripts\python.exe"
%PYEXE% -m pip install --upgrade pip
if errorlevel 1 goto install_fail
%PYEXE% -m pip install -r requirements.txt
if errorlevel 1 goto install_fail

:deps_ready
echo [INFO] Checking CUDA-enabled PyTorch...
%PYEXE% -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
%PYEXE% -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 goto cuda_fail

if /I "%~1"=="--smoke-test" goto smoke
if /I "%~1"=="--check-only" goto check_only
if /I "%~1"=="--memory-probe" goto memory_probe

echo [INFO] Running production preflight...
%PYEXE% run.py --check-only
if errorlevel 1 goto run_fail
echo [INFO] Starting the million-DOF optimization. Resume is enabled by default.
%PYEXE% run.py %*
if errorlevel 1 goto run_fail
goto success

:smoke
echo [INFO] Running the small pipeline test. Output is diagnostic only.
%PYEXE% run.py --smoke-test
if errorlevel 1 goto run_fail
goto success

:check_only
%PYEXE% run.py --check-only
if errorlevel 1 goto run_fail
goto success

:memory_probe
%PYEXE% memory_probe.py
if errorlevel 1 goto run_fail
goto success

:success
echo [SUCCESS] Finished. See the results directory and corresponding run.log for details.
if not defined NO_PAUSE pause
exit /b 0

:no_python
echo [ERROR] Native Windows Python 3.10+ was not found.
echo Install Python from python.org and enable pip, or set PYTHON to python.exe.
goto fail

:venv_fail
echo [ERROR] Native Windows venv creation failed.
echo Reinstall Python from python.org with pip/ensurepip enabled. Do not use WSL.
goto fail

:install_fail
echo [ERROR] Python package installation failed. Check the network and pip configuration.
goto fail

:cuda_fail
echo [ERROR] CUDA-enabled PyTorch is unavailable.
echo Install a GPU-compatible PyTorch build and check nvidia-smi. The production case will not use CPU fallback.
goto fail

:run_fail
echo [ERROR] Computation failed. Review the error above and results\million_1480x504\run.log.
goto fail

:fail
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" set "RC=1"
echo [FAILED] Exit code: %RC%
if not defined NO_PAUSE pause
exit /b %RC%
