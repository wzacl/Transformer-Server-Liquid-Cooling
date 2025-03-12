import threading
import time
import numpy as np


class SequenceWindow:
    def __init__(self, window_size=20, adams_controller=None):
        self.window_size = window_size
        self.buffer = np.zeros((window_size, len(adams_controller.buffer))) if adams_controller else np.zeros(window_size)
        self.adam = adams_controller
        self.flag_buffer = False
        self.buffer_lock = threading.Lock()
        self.th_buffer = threading.Thread(target=self.sequence_buffer)

    def sequence_buffer(self):
        while self.flag_buffer:
            try:
                if self.adam:
                    with self.buffer_lock:
                        features = [
                            self.adam.buffer[0],  # T_GPU
                            self.adam.buffer[2],  # T_CDU_in
                            self.adam.buffer[4],  # T_env
                            self.adam.buffer[5],  # T_air_in
                            self.adam.buffer[6],  # T_air_out
                            self.adam.buffer[8],  # fan_duty
                            self.adam.buffer[9]   # pump_duty
                        ]

                        new_data = np.array(features)
                        self.buffer = np.roll(self.buffer, -1, axis=0)
                        self.buffer[-1, :] = new_data  # 將新資料添加到buffer的末尾
                time.sleep(0.1)
            except Exception as e:
                print(f"Error reading from ADAMS controller: {e}")
        print('Sequence buffer thread stopped')

    def get_window_data(self):
        with self.buffer_lock:
            return list(self.buffer)
    
    def start_sequence_buffer(self):
        self.flag_buffer = True
        self.th_buffer.start()

    def stop_sequence_buffer(self):
        self.flag_buffer = False
        self.th_buffer.join()




