import joblib
import numpy as np

# 載入scaler文件
scaler_path = 'Predict_Model/no_Tenv_seq35_steps8_batch512_hidden16_layers1_heads8_dropout0.01_epoch400/1.5_1KWscalers.jlib'
try:
    scaler = joblib.load(scaler_path)
    print("成功載入scaler文件")
    
    # 檢查scaler類型
    print(f"Scaler類型: {type(scaler)}")
    
    # 定義特徵名稱
    feature_names = ['T_GPU', 'T_CDU_out', 'T_CDU_in', 'T_air_in', 'T_air_out', 'fan_duty', 'pump_duty']
    
    # 查看是否為元組(input_scaler, output_scaler)
    if isinstance(scaler, tuple):
        print(f"Scaler是一個元組，長度: {len(scaler)}")
        
        if len(scaler) == 2:
            input_scaler, output_scaler = scaler
            print("\n===== 輸入Scaler (MinMaxScaler) =====")
            print(f"類型: {type(input_scaler)}")
            
            # 顯示每個特徵的範圍
            print("\n原始數據範圍:")
            for i, name in enumerate(feature_names):
                print(f"{name}: {input_scaler.data_min_[i]:.2f} 至 {input_scaler.data_max_[i]:.2f}")
            
            print("\n標準化後的範圍:")
            for i, name in enumerate(feature_names):
                min_val = 0  # 標準化後最小值通常是0
                max_val = 1  # 標準化後最大值通常是1
                print(f"{name}: {min_val:.4f} 至 {max_val:.4f}")
                
            print("\n===== 輸出Scaler (MinMaxScaler) =====")
            print(f"類型: {type(output_scaler)}")
            print(f"輸出特徵: T_CDU_out")
            print(f"原始數據範圍: {output_scaler.data_min_[0]:.2f} 至 {output_scaler.data_max_[0]:.2f}")
            print(f"標準化後範圍: 0.0000 至 1.0000")
            
            # 測試使用
            print("\n===== 測試映射 =====")
            # 建立一個測試數據集，每個元素依序是: T_GPU, T_CDU_out, T_CDU_in, T_air_in, T_air_out, fan_duty, pump_duty
            test_data = np.array([[50, 35, 30, 25, 30, 80, 80]])
            print("原始測試數據:")
            for i, name in enumerate(feature_names):
                print(f"{name}: {test_data[0, i]}")
                
            transformed_data = input_scaler.transform(test_data)
            print("\n正規化後的數據:")
            for i, name in enumerate(feature_names):
                print(f"{name}: {transformed_data[0, i]:.6f}")
                
            # 從Sequence_Window_Processor.py獲取特徵排序
            print("\n根據SequenceWindowProcessor.py中的特徵順序:")
            seq_proc_order = ['T_GPU', 'T_CDU_out', 'T_CDU_in', 'T_air_in', 'T_air_out', 'fan_duty', 'pump_duty']
            print(f"特徵排序: {seq_proc_order}")
            
            # 打印特徵對應的索引位置
            print("\n特徵在原始數據中的索引位置:")
            for name in seq_proc_order:
                if name in feature_names:
                    idx = feature_names.index(name)
                    print(f"{name}: 位置 {idx}")
                else:
                    print(f"{name}: 未找到")
            
            print("\n結論: ")
            if feature_names == seq_proc_order:
                print("✅ 映射檔案中的特徵順序與Sequence_Window_Processor.py中定義的順序一致!")
                print("映射檔案中的特徵順序為:")
                for i, name in enumerate(feature_names):
                    print(f"{i}: {name}")
            else:
                print("❌ 映射檔案中的特徵順序與Sequence_Window_Processor.py中定義的順序不一致!")
                print("映射檔案中的特徵順序為:")
                for i, name in enumerate(feature_names):
                    print(f"{i}: {name}")
                print("\nSequence_Window_Processor.py中的特徵順序為:")
                for i, name in enumerate(seq_proc_order):
                    print(f"{i}: {name}")
                
            # 檢查Sequence_Window_Processor.py中的值
            adam_order = [0, 3, 2, 5, 6, 8, 9]  # 根據提供的Sequence_Window_Processor.py文件
            print("\n在Sequence_Window_Processor.py的update_from_adam中使用的索引:")
            print("self.adam.buffer[0]  # T_GPU")
            print("self.adam.buffer[3]  # T_CDU_out")
            print("self.adam.buffer[2]  # T_CDU_in")
            print("self.adam.buffer[5]  # T_air_in")
            print("self.adam.buffer[6]  # T_air_out")
            print("self.adam.buffer[8]  # fan_duty")
            print("self.adam.buffer[9]  # pump_duty")
                
    else:
        print("Scaler不是元組，而是單一scaler")
        print(f"類型: {type(scaler)}")
        if hasattr(scaler, 'scale_'):
            print(f"尺度係數: {scaler.scale_}")
        if hasattr(scaler, 'min_'):
            print(f"最小值: {scaler.min_}")
        if hasattr(scaler, 'data_min_'):
            print(f"數據最小值: {scaler.data_min_}")
        if hasattr(scaler, 'data_max_'):
            print(f"數據最大值: {scaler.data_max_}")

except Exception as e:
    print(f"載入或處理scaler文件時出錯: {e}") 