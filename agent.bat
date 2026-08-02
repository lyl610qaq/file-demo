@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "PROJECT_DIR=C:\Users\13211\Desktop\myproject\workspace-agent-take-home"
set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "APP_URL=http://127.0.0.1:8000"
set "CHECK_ONLY=0"
if /i "%~1"=="--check" set "CHECK_ONLY=1"

echo [1/5] 检查项目目录...
if not exist "%PROJECT_DIR%\pyproject.toml" goto :missing_project
cd /d "%PROJECT_DIR%" || goto :missing_project

echo [2/5] 检查 Python 虚拟环境...
if exist "%VENV_PY%" goto :check_existing_venv

call :find_python
if errorlevel 1 goto :python_missing
"%BASE_PY%" %BASE_PY_ARGS% -m venv ".venv"
if errorlevel 1 goto :venv_failed
goto :environment_ready

:check_existing_venv
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 goto :broken_venv

:environment_ready
echo [3/5] 检查项目依赖...
"%VENV_PY%" -c "import fastapi, httpx, pydantic_settings, uvicorn, workspace_agent.web" >nul 2>&1
if errorlevel 1 (
    "%VENV_PY%" -m pip install -e .
    if errorlevel 1 goto :dependency_failed
)

echo [4/5] 检查本地配置...
if not exist ".env" (
    if not exist ".env.example" goto :env_example_missing
    copy /Y ".env.example" ".env" >nul
    if errorlevel 1 goto :env_copy_failed
    echo 已从 .env.example 创建 .env，请按需检查 API Key。
)

for /f "usebackq tokens=1,* delims==" %%A in (`findstr /B /C:"ALLOWED_ORIGIN=" ".env"`) do set "APP_URL=%%B"
set "APP_URL=%APP_URL:"=%"

if "%CHECK_ONLY%"=="1" (
    echo [5/5] 预检通过。启动地址：%APP_URL%
    exit /b 0
)

netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul
if not errorlevel 1 (
    echo 服务已在运行，正在打开 %APP_URL%
    start "" "%APP_URL%"
    exit /b 0
)

echo [5/5] 正在启动 Workspace Agent：%APP_URL%
start "" "%APP_URL%"
"%VENV_PY%" -m uvicorn workspace_agent.web:app --host 127.0.0.1 --port 8000
set "SERVER_EXIT=%ERRORLEVEL%"
if not "%SERVER_EXIT%"=="0" (
    echo.
    echo [错误] 服务异常退出，错误码：%SERVER_EXIT%
    pause
)
exit /b %SERVER_EXIT%

:find_python
set "BASE_PY="
set "BASE_PY_ARGS="

if not exist "%LocalAppData%\Programs\Python\Python313\python.exe" goto :try_local_python_312
set "BASE_PY=%LocalAppData%\Programs\Python\Python313\python.exe"
call :validate_python
if not errorlevel 1 exit /b 0

:try_local_python_312
if not exist "%LocalAppData%\Programs\Python\Python312\python.exe" goto :try_py_launcher
set "BASE_PY=%LocalAppData%\Programs\Python\Python312\python.exe"
call :validate_python
if not errorlevel 1 exit /b 0

:try_py_launcher
where py >nul 2>&1
if errorlevel 1 goto :try_path_python
py -3.13 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "BASE_PY=py"
    set "BASE_PY_ARGS=-3.13"
    exit /b 0
)
py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "BASE_PY=py"
    set "BASE_PY_ARGS=-3.12"
    exit /b 0
)

:try_path_python
where python >nul 2>&1
if errorlevel 1 exit /b 1
set "BASE_PY=python"

:validate_python
"%BASE_PY%" %BASE_PY_ARGS% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
exit /b %ERRORLEVEL%

:missing_project
echo [错误] 找不到项目：%PROJECT_DIR%
goto :failed

:python_missing
echo [错误] 未找到 Python 3.12 或更高版本。脚本不会自动下载 Python。
goto :failed

:broken_venv
echo [错误] 现有 .venv 无法运行或 Python 版本低于 3.12。
echo 请删除 %PROJECT_DIR%\.venv 后重新运行本脚本。
goto :failed

:venv_failed
echo [错误] 创建虚拟环境失败。
goto :failed

:dependency_failed
echo [错误] 安装项目依赖失败，请检查网络或 pip 输出。
goto :failed

:env_example_missing
echo [错误] 找不到 .env.example，无法创建首次配置。
goto :failed

:env_copy_failed
echo [错误] 创建 .env 失败，请检查目录权限。

:failed
if not "%CHECK_ONLY%"=="1" pause
exit /b 1
