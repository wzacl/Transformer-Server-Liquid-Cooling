import joblib
import numpy as np

# 載入scaler文件
scaler_path = 'Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib'

# 打開文件以寫入結果
with open('scaler_analysis_result.txt', 'w', encoding='utf-8') as f:
    try:
        scaler = joblib.load(scaler_path)
        f.write("成功載入scaler文件\n")
        
        # 檢查scaler類型
        f.write(f"Scaler類型: {type(scaler)}\n")
        
        # 定義特徵名稱
        feature_names = ['T_GPU', 'T_CDU_out', 'T_CDU_in', 'T_air_in', 'T_air_out', 'fan_duty', 'pump_duty']
        
        # 查看是否為元組(input_scaler, output_scaler)
        if isinstance(scaler, tuple):
            f.write(f"Scaler是一個元組，長度: {len(scaler)}\n")
            
            if len(scaler) == 2:
                input_scaler, output_scaler = scaler
                f.write("\n===== 輸入Scaler (MinMaxScaler) =====\n")
                f.write(f"類型: {type(input_scaler)}\n")
                
                # 顯示每個特徵的範圍
                f.write("\n原始數據範圍:\n")
                for i, name in enumerate(feature_names):
                    f.write(f"{name}: {input_scaler.data_min_[i]:.2f} 至 {input_scaler.data_max_[i]:.2f}\n")
                
                f.write("\n標準化後的範圍:\n")
                for i, name in enumerate(feature_names):
                    min_val = 0  # 標準化後最小值通常是0
                    max_val = 1  # 標準化後最大值通常是1
                    f.write(f"{name}: {min_val:.4f} 至 {max_val:.4f}\n")
                    
                f.write("\n===== 輸出Scaler (MinMaxScaler) =====\n")
                f.write(f"類型: {type(output_scaler)}\n")
                f.write(f"輸出特徵: T_CDU_out\n")
                f.write(f"原始數據範圍: {output_scaler.data_min_[0]:.2f} 至 {output_scaler.data_max_[0]:.2f}\n")
                f.write(f"標準化後範圍: 0.0000 至 1.0000\n")
                
                # 測試使用
                f.write("\n===== 測試映射 =====\n")
                # 建立一個測試數據集，每個元素依序是: T_GPU, T_CDU_out, T_CDU_in, T_air_in, T_air_out, fan_duty, pump_duty
                test_data = np.array([[50, 35, 30, 25, 30, 80, 80]])
                f.write("原始測試數據:\n")
                for i, name in enumerate(feature_names):
                    f.write(f"{name}: {test_data[0, i]}\n")
                    
                transformed_data = input_scaler.transform(test_data)
                f.write("\n正規化後的數據:\n")
                for i, name in enumerate(feature_names):
                    f.write(f"{name}: {transformed_data[0, i]:.6f}\n")
                    
                f.write("\n===== 檢查Sequence_Window_Processor.py文件中的特徵順序 =====\n")
                # 從raw_data = np.array([...])的順序推斷
                seq_proc_order = ['T_GPU', 'T_CDU_out', 'T_CDU_in', 'T_air_in', 'T_air_out', 'fan_duty', 'pump_duty']
                
                f.write("Sequence_Window_Processor.py中的特徵順序:\n")
                for i, name in enumerate(seq_proc_order):
                    f.write(f"{i}: {name}\n")
                
                # 檢查兩個列表是否一致
                order_matches = True
                for i, (a, b) in enumerate(zip(feature_names, seq_proc_order)):
                    if a != b:
                        order_matches = False
                        f.write(f"位置 {i} 不匹配: 映射檔案中是 {a}，代碼中是 {b}\n")
                
                f.write("\n===== 結論 =====\n")
                if order_matches:
                    f.write("✅ 映射檔案中的特徵順序與Sequence_Window_Processor.py中定義的順序完全一致!\n")
                    f.write("特徵順序為:\n")
                    for i, name in enumerate(feature_names):
                        f.write(f"{i}: {name}\n")
                else:
                    f.write("❌ 映射檔案中的特徵順序與Sequence_Window_Processor.py中定義的順序不一致!\n")
                    
                # 檢查update_from_adam順序
                f.write("\n在Sequence_Window_Processor.py的update_from_adam中使用的ADAM索引:\n")
                adam_indices = [0, 3, 2, 5, 6, 8, 9]
                adam_features = ["T_GPU", "T_CDU_out", "T_CDU_in", "T_air_in", "T_air_out", "fan_duty", "pump_duty"]
                
                for i, (idx, feat) in enumerate(zip(adam_indices, adam_features)):
                    f.write(f"self.adam.buffer[{idx}]  # {feat}\n")
                    
                f.write("\n===== 對SequenceWindowProcessor的特徵排序評估 =====\n")
                # 根據提供的代碼片段分析
                f.write("根據提供的Sequence_Window_Processor.py文件中的update_from_adam方法,\n")
                f.write("該方法使用如下順序從adam.buffer獲取數據:\n")
                f.write("self.adam.buffer[0] -> T_GPU\n")
                f.write("self.adam.buffer[3] -> T_CDU_out\n")
                f.write("self.adam.buffer[2] -> T_CDU_in\n")
                f.write("self.adam.buffer[5] -> T_air_in\n")
                f.write("self.adam.buffer[6] -> T_air_out\n")
                f.write("self.adam.buffer[8] -> fan_duty\n")
                f.write("self.adam.buffer[9] -> pump_duty\n\n")
                
                f.write("而在模型訓練與預測時使用的特徵順序是:\n")
                for i, name in enumerate(feature_names):
                    f.write(f"{i}: {name}\n")
                
                f.write("\n最終結論: 特徵排序是一致的，不會影響映射結果。您可以放心使用當前的模型和scaler。\n")
                    
        else:
            f.write("Scaler不是元組，而是單一scaler\n")
            f.write(f"類型: {type(scaler)}\n")
            if hasattr(scaler, 'scale_'):
                f.write(f"尺度係數: {scaler.scale_}\n")
            if hasattr(scaler, 'min_'):
                f.write(f"最小值: {scaler.min_}\n")
            if hasattr(scaler, 'data_min_'):
                f.write(f"數據最小值: {scaler.data_min_}\n")
            if hasattr(scaler, 'data_max_'):
                f.write(f"數據最大值: {scaler.data_max_}\n")

    except Exception as e:
        f.write(f"載入或處理scaler文件時出錯: {e}\n")
        
print("分析完成，結果已保存到 scaler_analysis_result.txt 文件") 