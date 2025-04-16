# 伺服器液冷系統路徑配置使用說明

## 簡介

本專案使用集中式路徑配置系統，解決在不同Linux系統上運行時的路徑引用問題。此配置系統的核心是：
1. 預設路徑配置（`config.py`）
2. 用戶本地配置（`local_config.py`）

通過這種方式，您可以在不修改源代碼的情況下，為不同環境設置特定的路徑。

## 使用方法

### 基本使用

大多數源代碼檔案已經更新，使用統一的配置引入方式：

```python
import sys
import os

# 添加專案根目錄到路徑以引入config模塊
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入配置並設置路徑
from config import setup_paths, get_path
setup_paths()

# 使用配置中的路徑
device_path = get_path('adam_port')
data_dir = get_path('data_dir')
```

### 自定義配置

若需要調整路徑配置：

1. 複製 `local_config_template.py` 為 `local_config.py`
2. 在 `local_config.py` 中修改 `LOCAL_PATHS` 字典，只需包含要覆蓋的路徑
3. 重新啟動應用程序

例如：
```python
# local_config.py
LOCAL_PATHS = {
    'adam_port': '/dev/ttyUSB1',  # 使用不同的ADAM端口
    'data_dir': '/mnt/shared/cooling_data',  # 使用共享存儲路徑
}
```

### 配置功能

`config.py` 模組提供以下功能：

1. **setup_paths()**: 將所有定義的路徑自動添加到 `sys.path`
2. **get_path(key)**: 獲取指定名稱的路徑
3. **print_paths()**: 打印所有當前配置的路徑

## 路徑說明

主要預設路徑包括：

| 路徑鍵名 | 描述 | 預設值 |
|---------|------|-------|
| project_root | 專案根目錄 | (自動檢測) |
| controllers | 控制器目錄 | PROJECT_ROOT/Controllers |
| control_unit | 控制單元目錄 | PROJECT_ROOT/Control_Unit |
| predict_model | 預測模型目錄 | PROJECT_ROOT/Predict_Model |
| data_dir | 數據目錄 | PROJECT_ROOT/data_manage |
| adam_port | ADAM設備端口 | /dev/ttyUSB0 |
| fan1_port | 風扇1設備端口 | /dev/ttyAMA4 |
| fan2_port | 風扇2設備端口 | /dev/ttyAMA5 |
| pump_port | 泵設備端口 | /dev/ttyAMA3 |

完整路徑列表請查看 `config.py` 中的 `DEFAULT_PATHS` 字典。

## 最佳實踐

1. **每台設備建立專屬配置**: 為每台運行設備創建專屬的 `local_config.py`
2. **不要提交local_config.py**: 確保 `local_config.py` 已在 `.gitignore` 中
3. **只覆蓋必要的路徑**: 本地配置只需包含與預設不同的路徑
4. **使用相對路徑**: 盡可能使用相對於 `project_root` 的路徑

## 示例

### 例1: 在不同Linux發行版上運行

Ubuntu機器上的配置:
```python
LOCAL_PATHS = {
    'adam_port': '/dev/ttyACM0',  # Ubuntu上的設備路徑可能不同
}
```

CentOS機器上的配置:
```python
LOCAL_PATHS = {
    'adam_port': '/dev/ttyS0',
    'data_dir': '/var/log/cooling_system',  # 使用不同的日誌目錄
}
```

### 例2: 使用共享存儲

```python
LOCAL_PATHS = {
    'data_dir': '/mnt/shared/cooling_data',
    'control_data': '/mnt/shared/cooling_data/control',
}
```

## 排錯

如果遇到路徑問題：

1. 運行 `python config.py` 來查看當前路徑配置
2. 檢查 `local_config.py` 中是否正確設置了路徑
3. 確保您修改的文件正確引入了配置系統
4. 檢查目錄和文件是否存在，權限是否正確

## 安全注意事項

避免在 `local_config.py` 中存儲敏感信息（如密碼）。此文件僅用於路徑配置，不應包含安全憑證或其他敏感數據。 