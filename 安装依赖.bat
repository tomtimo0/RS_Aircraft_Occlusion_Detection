@echo off
echo 遥感飞机检测APP 安装脚本
echo ================================
echo.

echo 正在检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python环境，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo.
echo 正在安装依赖包...
pip install -r requirements.txt
if errorlevel 1 (
    echo 错误: 依赖包安装失败
    pause
    exit /b 1
)

echo.
echo 安装完成！
echo 您现在可以运行 "启动APP.bat" 来启动应用
echo.
pause
