'''''
class XYKPWMController:
    def __init__(self, port, baudrate=9600, timeout=2):
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
        try:
            if command.lower() == 'end':
                self.close()
                return 'Goodbye'
            self.ser.write(command.encode())
            readOut = self.ser.readline().decode()
            return readOut
        except Exception as e:
            return str(e)

    def read_status(self):
        try:
            self.ser.write('READ'.encode())
            readOut = self.ser.readline().decode()
            return readOut
        except Exception as e:
            return str(e)

    def close(self):
        self.ser.close()
        sys.exit(0)


if __name__ == "__main__":
    uart_controller = XYKPWMController(port='/dev/ttyAMA1')
    command = ''
    while command.lower() != 'end':
        command = input('Enter Command: ')  
        if command.lower() == 'read':
            status = uart_controller.read_status()
            print("Status:", status)
        else:
            reply = uart_controller.send_command(command)
            print("Reply:", reply)
'''''
import time
import serial
import sys
'''''
以下程式碼經過處理，將輸入的命令用更簡單的方式呈現在使用者面前並非原始的通訊樣式
該pwm訊號產生器主要功能與對應的命令分別是：
1.設定頻率(F{frequency})，其中frequency為字串且始終保持三個字符，如:010
2.設定duty cycle(d{duty_cycle})其中duty_cycle為字串且始終保持三個字符，如:010
3.開啟pwm控制器("ON")
4.關閉pwm控制器("OFF")，但關閉這個功能要小心使用因為一旦關閉控制目標就會全速運轉需要注意
5.回傳值("READ")
'''''
class XYKPWMController:
    def __init__(self, port, baudrate=9600, timeout=2):
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
        return self.send_command(f"F{self.encode_frequency(frequency)}")

    def set_duty_cycle(self, duty_cycle):
        return self.send_command(f"d{self.encode_duty(duty_cycle)}")

    def set_frequency_and_duty(self, frequency, duty_cycle):
        freq_response = self.set_frequency(frequency)
        time.sleep(0.2)  # 等待一小段時間，確保第一個命令已被處理
        duty_response = self.set_duty_cycle(duty_cycle)
        return f"Frequency response: {freq_response}, Duty cycle response: {duty_response}"

    def turn_on(self):
        return self.send_command("ON")

    def turn_off(self):
        return self.send_command("OFF")

    def read_status(self):
        return self.send_command("READ")

    def close(self):
        self.ser.close()
        print("Serial port closed.")

    def exp_end(self):
       return self.send_command(f"d{'050'}") 
    
    def encode_duty(self, var):
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
    def encode_frequency(self,value):
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
