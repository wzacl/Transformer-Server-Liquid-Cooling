#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置模塊 - 集中管理系統路徑和設置

此模塊提供集中化的路徑管理，解決不同Linux系統中的路徑引用問題。
同時支持通過local_config.py進行本地化，實現設備路徑在不同環境中的靈活設置。

主要功能:
1. 定義所有系統使用的標準路徑
2. 提供簡潔的API獲取正確路徑
3. 支持從local_config.py加載本地配置
4. 提供路徑有效性檢查和目錄創建功能

使用方式:
    from config import get_path, setup_paths, print_paths
    
    # 獲取特定路徑
    data_dir = get_path('data_dir')
    
    # 確保所有必要目錄存在
    setup_paths()
    
    # 顯示當前配置的所有路徑
    print_paths()
"""

import os
import sys
import platform
import pprint
from pathlib import Path

# 確定項目根目錄
if getattr(sys, 'frozen', False):
    # 如果是打包后的可執行文件
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    # 如果是開發環境
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 根據操作系統決定設備端口命名
if platform.system() == 'Linux':
    DEFAULT_ADAM_PORT = '/dev/ttyUSB0'
    DEFAULT_FAN1_PORT = '/dev/ttyAMA4'
    DEFAULT_FAN2_PORT = '/dev/ttyAMA5'
    DEFAULT_PUMP_PORT = '/dev/ttyAMA3'
else:  # Windows
    DEFAULT_ADAM_PORT = 'COM1'
    DEFAULT_FAN1_PORT = 'COM2'
    DEFAULT_FAN2_PORT = 'COM3'
    DEFAULT_PUMP_PORT = 'COM4'

# 默認路徑配置
PATHS = {
    # 基礎目錄
    'project_root': PROJECT_ROOT,
    
    # 控制系統目錄
    'controllers': os.path.join(PROJECT_ROOT, 'controllers'),
    'control_unit': os.path.join(PROJECT_ROOT, 'control_unit'),
    'control_models': os.path.join(PROJECT_ROOT, 'control_models'),
    
    # 數據目錄
    'data_dir': os.path.join(PROJECT_ROOT, 'data'),
    'control_data': os.path.join(PROJECT_ROOT, 'data', 'control_data'),
    'model_data': os.path.join(PROJECT_ROOT, 'data', 'model_data'),
    'raw_data': os.path.join(PROJECT_ROOT, 'data', 'raw_data'),
    
    # 日誌目錄
    'logs_dir': os.path.join(PROJECT_ROOT, 'logs'),
    'control_logs': os.path.join(PROJECT_ROOT, 'logs', 'control_logs'),
    'system_logs': os.path.join(PROJECT_ROOT, 'logs', 'system_logs'),
    
    # Web界面目錄
    'web_gui': os.path.join(PROJECT_ROOT, 'WebGUI'),
    'web_static': os.path.join(PROJECT_ROOT, 'WebGUI', 'static'),
    'web_templates': os.path.join(PROJECT_ROOT, 'WebGUI', 'templates'),
    
    # 設備端口
    'adam_port': DEFAULT_ADAM_PORT,
    'fan1_port': DEFAULT_FAN1_PORT,
    'fan2_port': DEFAULT_FAN2_PORT,
    'pump_port': DEFAULT_PUMP_PORT,
}

# 嘗試從local_config.py加載本地配置
try:
    from local_config import LOCAL_PATHS
    print("已加載本地配置文件local_config.py")
    # 更新默認路徑配置
    for key, value in LOCAL_PATHS.items():
        # 處理相對路徑（轉換為絕對路徑）
        if isinstance(value, str) and value.startswith('./'):
            value = os.path.join(PROJECT_ROOT, value[2:])
        PATHS[key] = value
except ImportError:
    print("未找到local_config.py，使用默認配置")
    print(f"如需自定義配置，請複製local_config_template.py為local_config.py並修改")

def get_path(key):
    """
    獲取指定的路徑
    
    Args:
        key (str): 路徑鍵名
        
    Returns:
        str: 對應的路徑
        
    Raises:
        KeyError: 如果指定的鍵不存在
    """
    try:
        return PATHS[key]
    except KeyError:
        available_keys = ', '.join(sorted(PATHS.keys()))
        raise KeyError(f"配置中未定義路徑鍵 '{key}'。可用的鍵: {available_keys}")

def setup_paths():
    """
    確保所有必要的目錄都存在，如果不存在則創建
    
    Returns:
        list: 新創建的目錄列表
    """
    created_dirs = []
    
    # 需要確保存在的目錄類型
    dir_keys = [
        'controllers', 'control_unit', 'control_models',
        'data_dir', 'control_data', 'model_data', 'raw_data',
        'logs_dir', 'control_logs', 'system_logs',
        'web_gui', 'web_static', 'web_templates'
    ]
    
    for key in dir_keys:
        path = get_path(key)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            created_dirs.append(path)
            print(f"已創建目錄: {path}")
    
    return created_dirs

def print_paths():
    """
    分類顯示當前配置的所有路徑及其存在狀態
    """
    # 定義路徑類別
    path_categories = {
        '基礎目錄': ['project_root'],
        '控制系統': ['controllers', 'control_unit', 'control_models'],
        '數據目錄': ['data_dir', 'control_data', 'model_data', 'raw_data'],
        '日誌目錄': ['logs_dir', 'control_logs', 'system_logs'],
        'Web界面': ['web_gui', 'web_static', 'web_templates'],
        '設備端口': ['adam_port', 'fan1_port', 'fan2_port', 'pump_port']
    }
    
    print("\n=== 系統路徑配置 ===")
    
    for category, keys in path_categories.items():
        print(f"\n--- {category} ---")
        for key in keys:
            if key in PATHS:
                path = PATHS[key]
                # 檢查是否為設備端口
                if key.endswith('_port'):
                    status = "（端口配置）"
                # 檢查是否為目錄且存在
                elif os.path.isdir(path):
                    status = "（已存在）"
                # 檢查是否為文件且存在
                elif os.path.isfile(path):
                    status = "（文件）"
                # 路徑不存在
                else:
                    status = "（不存在）"
                print(f"{key}: {path} {status}")
            else:
                print(f"{key}: 未定義")
    
    # 顯示其他未分類的路徑
    other_keys = set(PATHS.keys()) - {k for keys in path_categories.values() for k in keys}
    if other_keys:
        print("\n--- 其他路徑 ---")
        for key in sorted(other_keys):
            path = PATHS[key]
            if os.path.exists(path):
                status = "（已存在）" if os.path.isdir(path) else "（文件）"
            else:
                status = "（不存在）"
            print(f"{key}: {path} {status}")

# 當此模塊被直接執行時，顯示所有路徑
if __name__ == "__main__":
    print("變換器服務器液冷系統 - 配置模塊")
    print(f"項目根目錄: {PROJECT_ROOT}")
    print(f"操作系統: {platform.system()} {platform.release()}")
    
    print_paths()
    
    print("\n使用示例:")
    print("from config import get_path")
    print("data_dir = get_path('data_dir')") 