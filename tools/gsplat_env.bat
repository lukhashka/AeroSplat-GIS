@echo off
rem Environment for JIT-building gsplat's CUDA extension on Windows.
rem Usage: tools\gsplat_env.bat <command...>   (e.g. venv\Scripts\python -m aerosplat.gs.trainer ...)
rem Why: nvcc 12.6 rejects VS 18 (cudafe++ crashes), so use VS 2022 BuildTools;
rem      CUDA_HOME must match torch's CUDA (12.6), not the system default.
rem One-time patch in venv\Lib\site-packages\gsplat\cuda\_backend.py:
rem      extra_cflags must not contain "-Wno-attributes" on Windows (cl rejects it).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set TORCH_CUDA_ARCH_LIST=8.9
%*
