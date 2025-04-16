# 伺服器液冷系統安裝指南

## 環境要求
- Python 3.8+
- CUDA 11.0+ (可選，用於GPU加速)
- Raspberry Pi OS 或 Linux 系統

## 安裝步驟

### 1. 安裝Miniconda (推薦)
```bash
# 下載Miniconda安裝腳本（ARM架構版本）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda.sh

# 增加執行權限
chmod +x ~/miniconda.sh

# 批次安裝模式
bash ~/miniconda.sh -b

# 設置環境變數
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 初始化conda
conda init bash
```

### 2. 創建虛擬環境
```bash
# 創建Python 3.9的環境
conda create -n Server_Cooling python=3.9

# 啟動環境
conda activate Server_Cooling
```

### 3. 安裝依賴套件
```bash
# 安裝Python依賴套件
pip install -r requirements.txt

# 如果使用GPU，安裝相應CUDA版本的PyTorch
# 請訪問 https://pytorch.org/get-started/locally/ 獲取適合您系統的安裝命令
```

### 4. 硬體連接設置
確保以下硬體連接正確：
- ADAM 模組：通常連接到 `/dev/ttyUSB0`
- 泵控制：通常連接到 `/dev/ttyAMA3`
- 風扇1：通常連接到 `/dev/ttyAMA4`
- 風扇2：通常連接到 `/dev/ttyAMA5`

若硬體連接與上述不同，請修改相關程式碼中的連接埠設置。

### 5. 權限設置
```bash
# 確保有串行埠訪問權限
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyAMA3
sudo chmod 666 /dev/ttyAMA4
sudo chmod 666 /dev/ttyAMA5

# 重新啟動系統以使權限生效
sudo reboot
```

## 問題排除

### 常見問題

1. **串行埠無法訪問**
   - 檢查硬體連接
   - 確認權限設置
   - 使用 `ls -l /dev/tty*` 檢查設備是否存在

2. **相依性衝突**
   - 確保在隔離的虛擬環境中安裝套件
   - 使用 `pip list` 檢查已安裝的套件版本

3. **硬體連接問題**
   - 檢查接線與端口
   - 使用 `dmesg | grep tty` 檢查系統識別的串行裝置

## 測試安裝

完成安裝後，可通過運行以下命令測試系統：

```bash
# 執行循環實驗
python data_collection/experiment_cycle.py

# 或執行預測模型訓練
python Model_Training/2KWCDU_Predictor_multi_v3.py
``` 