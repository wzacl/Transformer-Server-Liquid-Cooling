#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本地配置模板

此文件用作本地配置的模板，請複製為 local_config.py 並根據您的環境進行修改。
local_config.py 不會被版本控制系統追踪，可以安全地存儲特定於設備的配置。

使用方法:
1. 複製此文件: cp local_config_template.py local_config.py
2. 根據您的環境修改 local_config.py 中的設置
3. 系統會自動加載 local_config.py 中的設置覆蓋默認配置
"""

# 本地路徑配置，將覆蓋config.py中的默認設置
LOCAL_PATHS = {
    # 示例: 設備串口路徑（根據您的環境修改）
    'adam_port': '/dev/ttyUSB0',  # 控制板串口
    'fan1_port': '/dev/ttyAMA4',  # 風扇1串口
    'fan2_port': '/dev/ttyAMA5',  # 風扇2串口
    'pump_port': '/dev/ttyAMA3',  # 水泵串口
    
    # 示例: 使用相對路徑（以.開頭）
    # 'data_dir': './custom_data',  # 將解析為專案根目錄下的custom_data目錄
    
    # 示例: 使用絕對路徑
    # 'logs_dir': '/var/log/transformer_cooling',  # 日誌存儲在系統日誌目錄
    
    # 您可以添加自定義路徑
    # 'custom_path': '/path/to/custom/directory',
}

# 本地調試設置
DEBUG = True  # 設置為True啟用調試模式
LOG_LEVEL = 'DEBUG'  # 日誌級別: DEBUG, INFO, WARNING, ERROR, CRITICAL

# 控制系統參數
#CONTROL_PARAMS = {
#    'sampling_interval': 1.0,  # 採樣間隔（秒）
#    'control_interval': 5.0,   # 控制間隔（秒）
#}

# 其他配置項（根據需要添加）
# CUSTOM_CONFIG = {
#     'key1': 'value1',
#     'key2': 'value2',
# }

# 如果此文件被直接運行，顯示提示信息
if __name__ == "__main__":
    print("這是本地配置模板文件，請勿直接運行。")
    print("請將此文件複製為local_config.py並進行自定義設置。")
    print("示例: cp local_config_template.py local_config.py") 