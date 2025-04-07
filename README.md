# Transformer-Server-Liquid-Cooling

## 📌 專案簡介
本專案 `Transformer-Server-Liquid-Cooling` 利用 **Transformer 模型** 來預測並控制 **伺服器液冷系統** 的運作。  
透過即時數據收集、機器學習模型訓練以及 PID 控制器來 **最佳化冷卻效率**，以降低能耗並提高運行穩定性。

## 🔧 系統架構
### 硬體配置
- **電源供應器參數設置**：
  * 1KW：220V_8A
  * 1.5KW：285V_8A
  * 1.85KW：325V_8A

- **控制元件限制**：
  * 泵最低轉速：40% duty cycle
  * 風扇最低轉速：30% duty cycle

### 軟體架構
- **資料收集模組** (`data_collection/`)
  * 實驗循環控制
  * 隨機參數測試
  * 數據記錄與存儲

- **控制單元** (`Control_Unit/`)
  * ADAMS 控制器
  * PWM 控制器
  * 多通道控制器

- **預測模型** (`Predict_Model/`)
  * Transformer 模型
  * 序列預測
  * 模型優化

- **模型訓練** (`Model_Training/`)
  * 數據預處理
  * 模型訓練
  * 效能評估

## 💻 安裝指南
1. **環境需求**
```bash
Python 3.8+
CUDA 11.0+ (可選，用於GPU加速)
```

2. **依賴套件**
```bash
pip install -r requirements.txt
```

3. **硬體連接**
```
ADAM 模組：/dev/ttyUSB0
泵控制：/dev/ttyAMA3
風扇1：/dev/ttyAMA4
風扇2：/dev/ttyAMA5
```

## 🚀 使用方法
### 1. 數據收集
```bash
# 執行循環實驗
python data_collection/experiment_cycle.py

# 執行隨機參數實驗
python data_collection/experiment_random.py
```

### 2. 模型訓練
```bash
# 訓練 Transformer 模型
python Model_Training/2KWCDU_Predictor_multi_v3.py
```

### 3. 系統控制
```bash
# 啟動 MPC 控制器
python Controllers/MPC/fan_MPC.py
```

## 📊 效能指標
- **預測準確度**：平均誤差 < 1°C
- **控制穩定性**：溫度波動 < ±0.5°C
- **能源效率**：較傳統方案節能 15-20%

## 🔍 監控與調試
### 數據監控
- 即時溫度監控
- 系統狀態記錄
- 異常檢測

### 效能優化
- 模型參數調整
- 控制策略優化
- 系統效能分析

## 📝 注意事項
1. 確保硬體連接正確
2. 定期檢查數據記錄
3. 監控系統溫度變化
4. 注意控制參數範圍

## 🛠 故障排除
### 常見問題
1. 串口通訊失敗
   - 檢查設備連接
   - 確認權限設置
   - 驗證串口參數

2. 預測異常
   - 檢查數據品質
   - 驗證模型參數
   - 重新訓練模型

3. 控制不穩定
   - 調整 PID 參數
   - 檢查感測器數據
   - 優化控制策略

## 📈 未來展望
- 整合深度強化學習
- 優化預測準確度
- 提升能源效率
- 擴展應用場景
