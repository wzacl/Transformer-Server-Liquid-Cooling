#!/usr/bin/env python3
"""實驗模式測試腳本

此腳本用於測試冷卻系統控制器的實驗模式功能，實現目標溫度的週期性變化。
"""
import sys
import os
import time
from Controllers.MPC.cooling_system_controller import (
    HardwareConfig,
    ModelConfig,
    ControlParameters,
    CoolingSystemController
)

def print_menu():
    """顯示選項菜單"""
    print("\n===== 實驗模式控制 =====")
    print("1. 啟動實驗模式 (使用預設參數)")
    print("2. 啟動實驗模式 (自定義參數)")
    print("3. 停止實驗模式")
    print("4. 設置實驗參數")
    print("5. 手動更新目標溫度")
    print("6. 退出程序")
    print("========================")
    return input("請選擇操作 [1-6]: ")

def setup_controller():
    """設置並返回控制器實例"""
    print("初始化控制器...")
    
    # 配置硬體
    hardware_config = HardwareConfig(
        # 使用模擬值或根據實際配置調整
        adam_port='/dev/ttyUSB0',
        fan1_port='/dev/ttyAMA4',
        fan2_port='/dev/ttyAMA5',
        pump_port='/dev/ttyAMA3'
    )
    
    # 配置模型
    model_config = ModelConfig(
        # 使用模擬值或根據實際路徑調整
        scaler_path="path/to/scalers.jlib",
        model_path="path/to/model.pth",
        exp_name="experiment_test"
    )
    
    # 初始化控制參數
    control_params = ControlParameters(
        gpu_target=71,  # 初始GPU目標溫度
        target_temp=30  # 初始系統目標溫度
    )
    
    # 創建並返回控制器 (不啟動主循環)
    return CoolingSystemController(
        hardware_config,
        model_config,
        control_params
    )

def set_experiment_parameters(controller):
    """設置實驗參數"""
    print("\n設置實驗參數:")
    
    try:
        # 獲取週期
        period = int(input("請輸入溫度變化週期(秒) [60-600]: ") or "300")
        
        # 獲取GPU目標溫度列表
        gpu_input = input("請輸入GPU目標溫度列表, 以逗號分隔 [60-85]: ") or "70,72,75,73"
        gpu_targets = [float(x.strip()) for x in gpu_input.split(",")]
        
        # 獲取系統目標溫度列表
        sys_input = input("請輸入系統目標溫度列表, 以逗號分隔 [20-40]: ") or "28,30,32,30"
        system_targets = [float(x.strip()) for x in sys_input.split(",")]
        
        # 更新實驗模式參數
        controller.experiment_mode.set_parameters(
            period=period,
            gpu_targets=gpu_targets,
            system_targets=system_targets
        )
        
    except ValueError as e:
        print(f"\n❌ 輸入錯誤: {e}")

def update_target_temperature(controller):
    """手動更新目標溫度"""
    print("\n手動更新目標溫度:")
    
    try:
        # 獲取GPU目標溫度
        gpu_target = float(input("請輸入新的GPU目標溫度 [60-85]: ") or "71")
        
        # 獲取系統目標溫度
        system_target = float(input("請輸入新的系統目標溫度 [20-40]: ") or "30")
        
        # 更新目標溫度
        controller.control_params.update_targets(
            gpu_target=gpu_target,
            target_temp=system_target
        )
        
    except ValueError as e:
        print(f"\n❌ 輸入錯誤: {e}")

def main():
    """主函數"""
    print("歡迎使用冷卻系統實驗模式測試腳本")
    
    try:
        # 設置控制器
        controller = setup_controller()
        print("控制器已初始化完成")
        
        while True:
            choice = print_menu()
            
            if choice == '1':
                # 使用預設參數啟動實驗模式
                controller.start_experiment_mode()
                
            elif choice == '2':
                # 設置自定義參數並啟動實驗模式
                set_experiment_parameters(controller)
                controller.start_experiment_mode()
                
            elif choice == '3':
                # 停止實驗模式
                controller.stop_experiment_mode()
                
            elif choice == '4':
                # 設置實驗參數
                set_experiment_parameters(controller)
                
            elif choice == '5':
                # 手動更新目標溫度
                update_target_temperature(controller)
                
            elif choice == '6':
                # 退出程序
                print("\n程序即將退出...")
                break
                
            else:
                print("\n❌ 無效選擇，請重新輸入")
                
            # 等待一會兒以便觀察控制器輸出
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序已被手動停止")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
    finally:
        print("\n✅ 程序已結束")

if __name__ == "__main__":
    main() 