#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置系統測試腳本

此腳本測試路徑配置系統是否正常工作，並顯示當前的配置細節。
"""

import os
import sys
import inspect

def separator(title=None, width=60):
    """顯示帶有標題的分隔線"""
    if title:
        left = (width - len(title) - 2) // 2
        right = width - left - len(title) - 2
        print("=" * left + f" {title} " + "=" * right)
    else:
        print("=" * width)

# 顯示基本環境信息
separator("環境信息")
print(f"Python版本: {sys.version}")
print(f"當前工作目錄: {os.getcwd()}")
print(f"腳本所在目錄: {os.path.dirname(os.path.abspath(__file__))}")

# 導入配置模塊並測試配置功能
separator("配置測試")
try:
    # 導入配置模塊
    from config import setup_paths, get_path, PATHS, print_paths
    
    # 檢查local_config是否存在
    has_local_config = False
    try:
        import local_config
        has_local_config = True
        print(f"已檢測到本地配置 (local_config.py)")
        print(f"本地配置內容: {local_config.LOCAL_PATHS}")
    except ImportError:
        print(f"未檢測到本地配置，使用預設配置")
    
    # 顯示所有路徑
    print_paths()
    
    # 測試sys.path是否已正確設置
    separator("路徑設置測試")
    
    print("執行setup_paths()...")
    setup_paths()
    
    # 顯示已添加到sys.path的路徑
    print("\n已添加到sys.path的專案路徑:")
    project_paths = [p for p in sys.path if os.path.abspath(p).startswith(get_path('project_root'))]
    for i, path in enumerate(project_paths, 1):
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{i}. {path} [{status}]")
    
    # 測試路徑訪問功能
    separator("路徑查詢測試")
    
    test_paths = [
        'project_root', 
        'controllers', 
        'control_unit', 
        'mpc', 
        'adam_port'
    ]
    
    for key in test_paths:
        value = get_path(key)
        exists = os.path.exists(value) if key not in ['adam_port', 'fan1_port', 'fan2_port', 'pump_port'] else "N/A (設備路徑)"
        status = "✓" if exists is True else "✗" if exists is False else exists
        print(f"路徑 '{key}': {value} [{status}]")
    
    separator("測試完成")
    
    print("\n路徑配置系統測試成功！ 👍")
    if not has_local_config:
        print("\n提示: 如果需要自定義路徑，請複製 local_config_template.py 為 local_config.py 並進行修改")
    
except Exception as e:
    separator("錯誤")
    print(f"配置系統測試失敗: {e}")
    import traceback
    traceback.print_exc() 