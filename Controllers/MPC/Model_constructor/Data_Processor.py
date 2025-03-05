import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib

class Data_Processor:
    def __init__(self, scaler_path, device):
        self.scaler = joblib.load(scaler_path)
        self.device = device
        #self.figure_path = figure_path

    def get_current_features(self, data):
        """獲取當前時間步的特徵"""
        if len(data) != 7:
            raise ValueError(f"輸入數據必須包含7個特徵,當前數據長度為:{len(data)}")
        
        data_2d = np.array(data).reshape(1, -1)
        if isinstance(self.scaler, tuple):
            scaled_data = self.scaler[0].transform(data_2d)
        else:
            scaled_data = self.scaler.transform(data_2d)
        
        return scaled_data[0]

    def prepare_sequence_data(self, history_buffer):
        if len(history_buffer) != 20:
            print(f"歷史數據長度錯誤: {len(history_buffer)}，需要 20 個時間步")
            return None
        sequence = np.array(list(history_buffer))
        if sequence.shape != (20, 7):
            print(f"數據形狀錯誤: {sequence.shape}，需要 (20, 7)")
            return None
        return torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

    def inverse_transform_predictions(self, scaled_predictions):
        if len(scaled_predictions.shape) == 1:
            scaled_predictions = scaled_predictions.reshape(-1, 1)

        if isinstance(self.scaler, tuple):
            return self.scaler[1].inverse_transform(scaled_predictions)[:, 0]
        else:
            return self.scaler.inverse_transform(scaled_predictions)[:, 0]


'''''
    def plot_future_predictions_with_event_markers(self, df):
        """繪製溫度預測曲線，並在風扇與泵轉速變動時標記事件點"""
        df.columns = df.columns.str.strip()

        fig, ax1 = plt.subplots(figsize=(12, 6))
        df['elapsed_time'] = np.arange(len(df))

        temp_col = 'actual_temp(CDU_out)' if 'actual_temp(CDU_out)' in df.columns else 'actual_temp'
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

        ax1.plot(df['elapsed_time'], df[temp_col], 'ro-', label='Actual Temperature', linewidth=2, markersize=6)

        for i in range(len(df)):
            future_steps = range(i, min(i + 8, len(df)))
            future_values = df.iloc[i, 5:5+len(future_steps)]

            if len(future_steps) > 1:
                ax1.plot(df['elapsed_time'].iloc[list(future_steps)], future_values, 'o--', color=colors[i], alpha=0.7, markersize=5)

        ax1.set_xlabel('Elapsed Time (seconds)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Predictions with Event Markers')
        ax1.grid(True, linestyle='--', alpha=0.7)

        for i in range(1, len(df)):
            if df['fan_duty'].iloc[i] != df['fan_duty'].iloc[i - 1]:
                ax1.axvline(x=df['elapsed_time'].iloc[i], color='blue', linestyle='--', alpha=0.8, label='Fan Change' if 'Fan Change' not in ax1.get_legend_handles_labels()[1] else "")
                ax1.text(df['elapsed_time'].iloc[i], df[temp_col].iloc[i], 'Fan Change', color='blue', fontsize=9, rotation=45, verticalalignment='bottom')

            if df['pump_duty'].iloc[i] != df['pump_duty'].iloc[i - 1]:
                ax1.axvline(x=df['elapsed_time'].iloc[i], color='green', linestyle='--', alpha=0.8, label='Pump Change' if 'Pump Change' not in ax1.get_legend_handles_labels()[1] else "")
                ax1.text(df['elapsed_time'].iloc[i], df[temp_col].iloc[i], 'Pump Change', color='green', fontsize=9, rotation=45, verticalalignment='bottom')

        ax1.legend(loc='upper left', fontsize=9)
        plt.savefig(self.figure_path)
'''''
