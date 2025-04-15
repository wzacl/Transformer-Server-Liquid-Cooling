#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
伺服器液冷系統Web監控界面
使用Flask框架提供實時溫度和控制數據的可視化
"""

from flask import Flask, render_template, jsonify, request
import sys
import os
import time
import json
import threading
import random

# 添加專案根目錄到路徑中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 初始化 Flask 應用
app = Flask(__name__)

# 模擬數據類（在實際環境中將連結到真實的感測器數據）
class DataSimulator:
    """模擬數據提供器，用於開發和測試"""
    
    def __init__(self):
        """初始化模擬數據參數"""
        # 初始溫度值
        self.T_GPU = 70.0
        self.T_heater = 75.0
        self.T_CDU_in = 50.0
        self.T_CDU_out = 30.0
        self.T_env = 25.0
        self.T_air_in = 20.0
        self.T_air_out = 35.0
        
        # 初始設備工作參數
        self.fan_duty = 60
        self.pump_duty = 60
        
        # 目標值
        self.GPU_target = 71
        self.target_temp = 30
        
        # 歷史數據
        self.history = {
            'timestamps': [],
            'T_GPU': [],
            'T_CDU_out': [],
            'fan_duty': [],
            'pump_duty': []
        }
        
        # SA優化日誌
        self.sa_logs = []
        self.optimization_in_progress = False
        
    def update(self):
        """更新模擬數據"""
        # 隨機變化溫度值（模擬真實數據變化）
        self.T_GPU += random.uniform(-0.5, 0.5)
        self.T_heater += random.uniform(-0.3, 0.3)
        self.T_CDU_in += random.uniform(-0.2, 0.2)
        self.T_CDU_out += random.uniform(-0.3, 0.3)
        self.T_env += random.uniform(-0.1, 0.1)
        self.T_air_in += random.uniform(-0.2, 0.2)
        self.T_air_out += random.uniform(-0.3, 0.3)
        
        # 保持溫度在合理範圍
        self.T_GPU = max(min(self.T_GPU, 85), 50)
        self.T_heater = max(min(self.T_heater, 90), 60)
        self.T_CDU_in = max(min(self.T_CDU_in, 60), 40)
        self.T_CDU_out = max(min(self.T_CDU_out, 40), 25)
        self.T_env = max(min(self.T_env, 30), 20)
        self.T_air_in = max(min(self.T_air_in, 30), 15)
        self.T_air_out = max(min(self.T_air_out, 45), 25)
        
        # 更新控制數值
        # 基於溫度調整泵和風扇轉速
        temp_diff_gpu = self.GPU_target - self.T_GPU
        temp_diff_cdu = self.target_temp - self.T_CDU_out
        
        # 模擬控制決策
        if temp_diff_gpu < -1:  # GPU溫度高於目標
            self.pump_duty = min(self.pump_duty + 5, 100)
        elif temp_diff_gpu > 1:  # GPU溫度低於目標
            self.pump_duty = max(self.pump_duty - 5, 40)
            
        if temp_diff_cdu < -1:  # 冷卻水溫度高於目標
            self.fan_duty = min(self.fan_duty + 5, 100)
        elif temp_diff_cdu > 1:  # 冷卻水溫度低於目標
            self.fan_duty = max(self.fan_duty - 5, 30)
        
        # 添加到歷史數據中
        timestamp = time.strftime('%H:%M:%S')
        if len(self.history['timestamps']) > 50:  # 保留最近的50筆數據
            for key in self.history:
                self.history[key].pop(0)
                
        self.history['timestamps'].append(timestamp)
        self.history['T_GPU'].append(round(self.T_GPU, 1))
        self.history['T_CDU_out'].append(round(self.T_CDU_out, 1))
        self.history['fan_duty'].append(self.fan_duty)
        self.history['pump_duty'].append(self.pump_duty)
        
    def start_optimization(self):
        """模擬SA優化過程"""
        if self.optimization_in_progress:
            return
            
        self.optimization_in_progress = True
        self.sa_logs = []
        
        # 清空優化日誌
        self.sa_logs.append("初始解：風扇轉速 = {}%, 預測成本 = {:.2f}".format(
            self.fan_duty, random.uniform(10, 20)))
        
        # 模擬優化迭代過程
        for i in range(5):
            # 模擬優化決策
            new_fan_duty = max(min(self.fan_duty + random.randint(-10, 10), 100), 30)
            temperature = 100 * (0.8 ** i)
            cost = random.uniform(10, 20)
            
            if random.random() < 0.7:  # 70%機率接受
                accept = True
                status = "接受"
            else:
                accept = False
                status = "拒絕"
                
            self.sa_logs.append("嘗試解：風扇轉速 = {}%, 預測成本 = {:.2f}, {} (溫度={:.2f})".format(
                new_fan_duty, cost, status, temperature))
                
            if accept:
                self.fan_duty = new_fan_duty
                if random.random() < 0.5:  # 50%機率找到更好的解
                    self.sa_logs.append("發現更好的解：風扇轉速 = {}%, 預測成本 = {:.2f}".format(
                        self.fan_duty, cost))
        
        # 添加最終結果
        self.sa_logs.append("最終解：風扇轉速 = {}%, 預測成本 = {:.2f}".format(
            self.fan_duty, random.uniform(8, 15)))
        self.sa_logs.append("最佳化完成，耗時 {:.2f} 秒".format(random.uniform(0.5, 2.0)))
        
        # 模擬優化時間
        time.sleep(random.uniform(1, 3))
        self.optimization_in_progress = False
        
    def update_target(self, gpu_target=None, cdu_target=None):
        """更新目標溫度"""
        if gpu_target is not None:
            self.GPU_target = float(gpu_target)
        if cdu_target is not None:
            self.target_temp = float(cdu_target)
        return {'GPU_target': self.GPU_target, 'target_temp': self.target_temp}

# 創建模擬數據提供實例
simulator = DataSimulator()

# 設置更新線程
def update_data():
    """背景線程：定期更新數據"""
    while True:
        simulator.update()
        time.sleep(1)  # 每秒更新一次

# 啟動更新線程
update_thread = threading.Thread(target=update_data)
update_thread.daemon = True
update_thread.start()

# 設置SA優化線程
def run_optimization():
    """背景線程：定期執行SA優化"""
    while True:
        simulator.start_optimization()
        time.sleep(20)  # 每20秒優化一次

# 啟動優化線程
optimization_thread = threading.Thread(target=run_optimization)
optimization_thread.daemon = True
optimization_thread.start()

@app.route('/')
def index():
    """渲染主頁面"""
    return render_template('index.html')

@app.route('/data')
def get_data():
    """API端點：獲取最新的溫度和控制數據"""
    data = {
        'temperatures': {
            'T_GPU': round(simulator.T_GPU, 1),
            'T_heater': round(simulator.T_heater, 1),
            'T_CDU_in': round(simulator.T_CDU_in, 1),
            'T_CDU_out': round(simulator.T_CDU_out, 1),
            'T_env': round(simulator.T_env, 1),
            'T_air_in': round(simulator.T_air_in, 1),
            'T_air_out': round(simulator.T_air_out, 1)
        },
        'controls': {
            'fan_duty': simulator.fan_duty,
            'pump_duty': simulator.pump_duty
        },
        'targets': {
            'GPU_target': simulator.GPU_target,
            'target_temp': simulator.target_temp
        },
        'optimization_status': simulator.optimization_in_progress
    }
    return jsonify(data)

@app.route('/history')
def get_history():
    """API端點：獲取歷史數據"""
    return jsonify(simulator.history)

@app.route('/logs')
def get_logs():
    """API端點：獲取SA優化日誌"""
    return jsonify({'logs': simulator.sa_logs})

@app.route('/update_target', methods=['POST'])
def update_target():
    """API端點：更新目標溫度"""
    data = request.json
    gpu_target = data.get('GPU_target')
    target_temp = data.get('target_temp')
    return jsonify(simulator.update_target(gpu_target, target_temp))

if __name__ == '__main__':
    try:
        print("伺服器液冷系統 Web 監控已啟動")
        print("訪問 http://127.0.0.1:5000 查看監控界面")
        app.run(debug=True, host='0.0.0.0')
    except KeyboardInterrupt:
        print("系統已停止") 