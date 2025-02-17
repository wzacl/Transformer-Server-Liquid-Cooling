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
class multichannel_PWMController:
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
        return self.send_command(f"F{frequency}")

    def set_duty_cycle(self, channel,duty_cycle):
        return self.send_command(f"D{channel}:{duty_cycle}")
    
    def set_frequency_and_duty(self, frequency, duty_cycle):

        return self.send_command(f"F{frequency}D{duty_cycle}")
    
    def set_all_duty_cycle(self,duty_cycle):
        for i in range(1,4):
            self.send_command(f"D{i}:{duty_cycle}")




    def read_status(self):
        return self.send_command("READ")

    def close(self):
        self.ser.close()
        print("Serial port closed.")


    def encode_frequency(self,value):
        try:
            if not 1 <= value <= 100:
                raise ValueError(f'Frequency must be between 1 and 100 kHz. Got {value}')

            # 創建映射字典
            frequency_map = {
                1: '1.00', 2: '2.00', 3: '3.00', 4: '4.00', 5: '5.00',
                6: '6.00', 7: '7.00', 8: '8.00', 9: '9.00', 10: '10.0',
                11: '11.0', 12: '12.0', 13: '13.0', 14: '14.0', 15: '15.0',
                16: '16.0', 17: '17.0', 18: '18.0', 19: '19.0', 20: '20.0',
                21: '21.0', 22: '22.0', 23: '23.0', 24: '24.0', 25: '25.0',
                26: '26.0', 27: '27.0', 28: '28.0', 29: '29.0', 30: '30.0'
            }

            if 1 <= value <= 30:
                # 對於1到30的值，使用映射
                rounded_value = round(value)
                return frequency_map[rounded_value]
            else:
                # 對於31到100的值，格式化為一位小數
                return f"{value:.1f}"
        except ValueError as e:
            print(f'Invalid input:{e}')
            return None
        except KeyError:
            print(f"Unexpected error: Value {rounded_value} not found in mapping")
            return None
if __name__ == "__main__":
    uart_controller = multichannel_PWMController(port='/dev/ttyAMA5')
    
    while True:
        print("\nAvailable commands:")
        print("1. Set Frequency")
        print("2. Set Duty Cycle")
        print("3. Set Frequency and Duty Cycle")
        print("4. Read Status")
        print("5. set all fan_duty")
        print("6. EXit")
        
        choice = input("Enter your choice (1-6): ")
        

        if choice == '1':
            freq = input("Enter frequency: ")
            print(uart_controller.set_frequency(freq))
        elif choice == '2':
            channel= input('Enter channel:')
            duty = input("Enter duty cycle: ")
            print(uart_controller.set_duty_cycle(channel,duty))
        elif choice == '3':
            freq = input("Enter frequency: ")
            duty = input("Enter duty cycle: ")
            print(uart_controller.set_frequency_and_duty(freq, duty))
        elif choice == '4':
            print(uart_controller.read_status())
        elif choice == '5':
            duty=input("Enter duty cycle: ")
            uart_controller.set_all_duty_cycle(duty)
        elif choice == '6':
            uart_controller.close()
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

