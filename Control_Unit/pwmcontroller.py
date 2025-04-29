import time
import serial
import sys

class XYKPWMController:
    """
    XYKPWM 控制器類，用於通過串口控制 PWM 設備。
    
    該類提供了與 XYKPWM 設備通信的方法，包括設置頻率、工作週期、開關控制等功能。
    
    Attributes:
        ser (serial.Serial): 串口通信對象，用於與 PWM 設備通信。
    """
    def __init__(self, port, baudrate=9600, timeout=2):
        """
        初始化 XYKPWM 控制器。
        
        Args:
            port (str): 串口設備路徑，例如 '/dev/ttyAMA3'。
            baudrate (int, optional): 波特率，默認為 9600。
            timeout (int, optional): 串口讀取超時時間（秒），默認為 2。
        """
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=timeout
        )
        print('Welcome to XYKPWM Controller')


    def send_command(self, command):
        """
        向 PWM 設備發送命令。
        
        Args:
            command (str): 要發送的命令字符串。
            
        Returns:
            str: 設備的響應或錯誤信息。
        """
        try:
            if command.lower() == 'end':
                self.close()
                return 'Goodbye'
            self.ser.write(command.encode())
            readOut = self.ser.readline().decode()
            return readOut
        except Exception as e:
            return str(e)

    def set_frequency(self, frequency):
        """
        設置 PWM 頻率。
        
        Args:
            frequency (str or int): 要設置的頻率值。
            
        Returns:
            str: 設備的響應。
        """
        return self.send_command(f"F{frequency}")

    def set_duty_cycle(self, duty_cycle):
        """
        設置 PWM 工作週期。
        
        Args:
            duty_cycle (str, int, or float): 工作週期值（0-100）。
            
        Returns:
            str: 設備的響應。
        """
        return self.send_command(f"d{self.encode_duty(duty_cycle)}")

    def set_frequency_and_duty(self, frequency, duty_cycle):
        """
        同時設置 PWM 頻率和工作週期。
        
        Args:
            frequency (str or int): 要設置的頻率值。
            duty_cycle (str, int, or float): 工作週期值（0-100）。
            
        Returns:
            str: 包含頻率和工作週期設置響應的字符串。
        """
        freq_response = self.set_frequency(self.encode_frequency(frequency))
        time.sleep(0.2)  # 等待一小段時間，確保第一個命令已被處理
        duty_response = self.set_duty_cycle(duty_cycle)
        return f"Frequency response: {freq_response}, Duty cycle response: {duty_response}"

    def turn_on(self):
        """
        打開 PWM 輸出。
        
        Returns:
            str: 設備的響應。
        """
        return self.send_command("ON")

    def turn_off(self):
        """
        關閉 PWM 輸出。
        
        Returns:
            str: 設備的響應。
        """
        return self.send_command("OFF")

    def read_status(self):
        """
        讀取 PWM 設備的當前狀態。
        
        Returns:
            str: 設備的狀態信息。
        """
        return self.send_command("READ")

    def close(self):
        """
        關閉串口連接。
        """
        self.ser.close()
        print("Serial port closed.")

    def exp_end(self):
        """
        實驗結束時將工作週期設置為 50%。
        
        Returns:
            str: 設備的響應。
        """
        return self.send_command(f"d{'050'}") 
    
    def encode_duty(self, var):
        """
        將工作週期值編碼為設備可接受的格式。
        
        Args:
            var (str, int, float, or list): 工作週期值或工作週期值列表。
            
        Returns:
            str or list: 編碼後的工作週期值或值列表。
            
        Raises:
            ValueError: 如果輸入值不在 0-100 範圍內。
            TypeError: 如果輸入類型不是數字或數字列表。
        """
        def encode_single(value):
            if isinstance(value, str):
                value = float(value)
            
            if not 0 <= value <= 100:
                raise ValueError(f"Invalid value: {value}. Must be between 0 and 100.")
            
            # 將 0-100 映射到 100-000
            encoded_value = int(value)
            return f"{encoded_value:03d}"  # 確保總是三位數

        if isinstance(var, (int, float, str)):
            return encode_single(var)
        elif isinstance(var, list):
            return [encode_single(v) for v in var]
        else:
            raise TypeError("Input must be a number or a list of numbers")
            
    def encode_frequency(self, value):
        """
        將頻率值編碼為設備可接受的格式。
        
        Args:
            value (int or float): 頻率值（1-30 kHz）。
            
        Returns:
            str: 編碼後的頻率值，或者在錯誤情況下返回 None。
            
        Raises:
            ValueError: 如果輸入值不在 1-30 範圍內。
        """
        try:
            if not 1<= value <=30:
                raise ValueError(f'Frequency must be between 1 and 30 kHz Got{value}')
             # 將值四捨五入到最接近的整數
            rounded_value=int(value)
        # 創建映射字典
            frequency_map = {
                1: '1.00', 2: '2.00', 3: '3.00', 4: '4.00', 5: '5.00',
                6: '6.00', 7: '7.00', 8: '8.00', 9: '9.00', 10: '10.0',
                11: '11.0', 12: '12.0', 13: '13.0', 14: '14.0', 15: '15.0',
                16: '16.0', 17: '17.0', 18: '18.0', 19: '19.0', 20: '20.0',
                21: '21.0', 22: '22.0', 23: '23.0', 24: '24.0', 25: '25.0',
                26: '26.0', 27: '27.0', 28: '28.0', 29: '29.0', 30: '30.0'
            }
            set_value=frequency_map[rounded_value]
            return set_value
        except ValueError as e:
            print(f'Invalid input:{e}')
            return None
        except KeyError:
            print(f"Unexpected error: Value {rounded_value} not found in mapping")
            return None

if __name__ == "__main__":
    # 當腳本直接運行時執行的代碼
    uart_controller = XYKPWMController(port='/dev/ttyAMA3')
    
    while True:
        print("\nAvailable commands:")
        print("1. Turn ON")
        print("2. Turn OFF")
        print("3. Set Frequency")
        print("4. Set Duty Cycle")
        print("5. Set Frequency and Duty Cycle")
        print("6. Read Status")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            print(uart_controller.turn_on())
        elif choice == '2':
            print(uart_controller.turn_off())
        elif choice == '3':
            freq = input("Enter frequency: ")
            print(uart_controller.set_frequency(freq))
        elif choice == '4':
            duty = input("Enter duty cycle: ")
            print(uart_controller.set_duty_cycle(duty))
        elif choice == '5':
            freq = input("Enter frequency: ")
            duty = input("Enter duty cycle: ")
            print(uart_controller.set_frequency_and_duty(freq, duty))
        elif choice == '6':
            print(uart_controller.read_status())
        elif choice == '7':
            uart_controller.close()
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

        time.sleep(0.5)  # 短暫暫停，以確保設備有時間處理命令
