#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用PyInstaller打包遥感飞机检测APP为可执行文件
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """安装PyInstaller"""
    print("正在安装PyInstaller...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("PyInstaller安装成功！")
        return True
    except subprocess.CalledProcessError:
        print("PyInstaller安装失败！")
        return False

def create_spec_file():
    """创建PyInstaller的spec文件"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 添加数据文件
added_files = [
    ('config.yaml', '.'),
    ('models_yolo/*.pt', 'models_yolo'),
    ('src/*.py', 'src'),
    ('requirements.txt', '.'),
    ('README.md', '.'),
]

a = Analysis(
    ['app_streamlit.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'streamlit',
        'streamlit_image_coordinates',
        'ultralytics',
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        'yaml',
        'src.image_utils',
        'src.yolo_detector',
        'src.visualization',
        'src.metrics',
        'src.main_cli',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='遥感飞机检测APP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='遥感飞机检测APP',
)
'''
    
    with open('app.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    print("已创建app.spec文件")

def build_executable():
    """构建可执行文件"""
    print("开始构建可执行文件...")
    
    # 检查是否存在spec文件
    if not os.path.exists('app.spec'):
        create_spec_file()
    
    try:
        # 使用PyInstaller构建
        cmd = [sys.executable, "-m", "PyInstaller", "app.spec", "--clean"]
        subprocess.check_call(cmd)
        print("可执行文件构建成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"构建失败: {e}")
        return False

def create_launcher_script():
    """创建启动脚本"""
    launcher_content = '''@echo off
echo 正在启动遥感飞机检测APP...
echo.
echo 请确保您已经安装了Python和必要的依赖包。
echo 如果没有安装，请运行: pip install -r requirements.txt
echo.
echo 启动中...
streamlit run app_streamlit.py
pause
'''
    
    with open('启动APP.bat', 'w', encoding='gbk') as f:
        f.write(launcher_content)
    print("已创建启动脚本: 启动APP.bat")

def create_install_script():
    """创建安装脚本"""
    install_content = '''@echo off
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
'''
    
    with open('安装依赖.bat', 'w', encoding='gbk') as f:
        f.write(install_content)
    print("已创建安装脚本: 安装依赖.bat")

def create_package():
    """创建完整的打包文件"""
    print("正在创建打包文件...")
    
    # 创建dist目录结构
    dist_dir = Path("dist/遥感飞机检测APP")
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制必要文件
    files_to_copy = [
        "app_streamlit.py",
        "config.yaml", 
        "requirements.txt",
        "README.md",
        "启动APP.bat",
        "安装依赖.bat"
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, dist_dir)
            print(f"已复制: {file}")
    
    # 复制src目录
    if os.path.exists("src"):
        shutil.copytree("src", dist_dir / "src", dirs_exist_ok=True)
        print("已复制: src目录")
    
    # 复制models_yolo目录（如果存在）
    if os.path.exists("models_yolo"):
        shutil.copytree("models_yolo", dist_dir / "models_yolo", dirs_exist_ok=True)
        print("已复制: models_yolo目录")
    
    # 创建使用说明
    readme_content = '''# 遥感飞机检测APP 使用说明

## 快速开始

1. **首次使用**：双击运行 "安装依赖.bat" 安装必要的Python包
2. **启动应用**：双击运行 "启动APP.bat" 启动应用
3. **浏览器访问**：应用启动后会自动打开浏览器，或手动访问 http://localhost:8501

## 功能说明

- 上传遥感图像和标签文件
- 选择多个YOLO模型进行对比检测
- 模拟高斯光斑遮挡效果
- 计算OOAP（遮挡面积百分比）指标
- 可视化检测结果和真值标注

## 系统要求

- Windows 10/11
- Python 3.8或更高版本
- 至少4GB内存
- 支持CUDA的显卡（可选，用于加速推理）

## 故障排除

如果遇到问题，请检查：
1. Python环境是否正确安装
2. 依赖包是否完整安装
3. 模型文件是否存在于models_yolo目录
4. 防火墙是否阻止了应用访问网络

## 技术支持

如有问题，请联系开发团队。
'''
    
    with open(dist_dir / "使用说明.txt", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n打包完成！文件位于: {dist_dir}")
    print("您可以将整个文件夹分发给其他用户使用。")

def main():
    """主函数"""
    print("遥感飞机检测APP 打包工具")
    print("=" * 50)
    
    # 检查PyInstaller
    try:
        import PyInstaller
        print("PyInstaller已安装")
    except ImportError:
        if not install_pyinstaller():
            return
    
    # 创建启动脚本
    create_launcher_script()
    create_install_script()
    
    # 询问用户选择打包方式
    print("\n请选择打包方式:")
    print("1. 创建可执行文件 (需要PyInstaller)")
    print("2. 创建便携式包 (推荐)")
    print("3. 两种方式都创建")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        if build_executable():
            print("可执行文件构建完成！")
        else:
            print("可执行文件构建失败，但便携式包仍可创建。")
    
    if choice in ['2', '3']:
        create_package()
    
    print("\n打包过程完成！")

if __name__ == "__main__":
    main() 