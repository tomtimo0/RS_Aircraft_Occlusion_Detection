#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遥感飞机检测APP打包配置
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "遥感飞机检测与高斯光斑模拟APP"

# 读取requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="remote-sensing-detection-app",
    version="1.3.0",
    author="遥感检测团队",
    author_email="team@example.com",
    description="遥感图像飞机目标检测与高斯光斑模拟应用",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/RemoteSensingObjectDetectionApp",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.txt', '*.md'],
    },
    install_requires=read_requirements(),
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    entry_points={
        'console_scripts': [
            'remote-sensing-app=app_streamlit:main',
        ],
    },
    keywords="remote sensing, object detection, YOLO, computer vision, AI",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/RemoteSensingObjectDetectionApp/issues",
        "Source": "https://github.com/your-repo/RemoteSensingObjectDetectionApp",
    },
) 