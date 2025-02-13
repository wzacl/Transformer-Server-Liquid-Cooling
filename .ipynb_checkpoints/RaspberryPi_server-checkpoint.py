import socket
import threading
import time
import ADAMScontroller
import pwmcontroller as ctrl
import multi_channel_pwmcontroller as multi_ctrl

        
class CDUServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        
        #這邊要輸入的參數分別為adam4118、xykpwm所在的port,可以先執行同資料夾內的port_check.py檔案查看
        self.adam_port = '/dev/ttyUSB0' 
        self.fan1_port='/dev/ttyAMA0'
        self.fan2_port='/dev/ttyAMA2'
        self.pump_port='/dev/ttyAMA3'
        self.exp_name = '2024.8.15測試'#該參數通常為實驗日期或是要改變參數的大方向 為資料夾名稱
        self.exp_var = '伺服器連線測試'#該參數為數據存入的檔案名稱，檔案讀取到的溫度數據最終會被存成csv檔案
        self.adam = None
        self.fan1=multi_ctrl.multichannel_PWMController(self.fan1_port)
        self.fan2=multi_ctrl.multichannel_PWMController(self.fan2_port)
        self.pump=ctrl.XYKPWMController(self.pump_port)
        self.running = True
        self.temperatures = [0] * 10  # 初始化10個溫度值
        self.socket_timeout = 1  # 1 秒超時

    def adam_activate(self):
        print('ADAM4118 開始工作')
        self.adam = ADAMScontroller.DataAcquisition(self.exp_name, self.exp_var, port=self.adam_port)
        self.adam.setup_directories()
        self.adam.start_data_buffer()
        self.adam.start_adam_controller()

    def update_temperatures(self):
        while self.running:
            with self.adam.buffer_lock:
                self.temperatures = self.adam.buffer.tolist()  # 將所有溫度值複製到我們的列表中
            time.sleep(3)  # 每秒更新一次溫度
 
 #---------------------------------處理輸入命令的函數-------------------------------------------
 #以下函數會負責處理來自客戶端(個人電腦)的命令，其功能包含讀取adam的資料、建立連線、斷開連線、更改duty cycle的大小
    def handle_client(self, client_socket):
        client_address=client_socket.getpeername()
        try:
            while self.running:
                try:
                    data = client_socket.recv(1024).decode('utf-8')
                    if not data:
                        print(f"Client {client_address} disconnected")
                        break  # 客戶端已斷開連接

                    if data == 'GET_TEMP_TABLE':#這會回傳一個溫度是初期用來Debug用，現在用處不大
                        response = "Current temperatures:\n"
                        for i, temp in enumerate(self.temperatures):
                            response += f"Channel {i}: {temp:.2f}°C\n"
                        client_socket.send(response.encode('utf-8'))
                    elif data == 'GET_TEMP':
                        buffer_data = ','.join(f"{temp:.2f}" for temp in self.adam.buffer if temp != 0)
                        if not buffer_data:
                            buffer_data = "0.00"  # 如果所有值都是0，至少發送一個0
                        client_socket.send(buffer_data.encode('utf-8'))
                    elif data == 'QUIT':
                        break
                    elif data == 'DISCONNECT':
                        print(f"Client {client_address} requested disconnect")
                        break 

                    elif data.startswith('set fan duty:'):
                        try:
                            # 拆分命令，格式應為 "set fan duty: fan_number:duty"
                            _, fan_info = data.split(':', 1)
                            fan_number, duty = map(int, fan_info.split(':'))
                            
                            
                            # 檢查風扇號碼和duty是否在有效範圍內
                            if fan_number in [1, 2, 3] and 0 <= duty <= 100:
                                # 假設self.fans是一個包含三個風扇對象的列表
                                self.fan1.set_duty_cycle(fan_number,duty)
                                response = f"Fan {fan_number} duty cycle set to {duty}"
                            elif fan_number in [4,5,6] and 0<=duty<=100:
                                number=fan_number-3
                                self.fan2.set_duty_cycle(number,duty)
                            else:
                                response = "Duty cycle must be between 0 and 100"
                        except ValueError:
                            response = "Invalid fan number or duty cycle value"
                        except IndexError:
                            response = "Fan number out of range"
                        except Exception as e:
                            response = f"Error setting duty cycle: {str(e)}"
                        
                        client_socket.send(response.encode('utf-8'))

                    elif data.startswith('set all fans duty: '):
                        try:
                            duty = float(data.split(':')[1])
                            duty=int(duty)
                            self.fan1.set_duty_cycle(1,duty)
                            self.fan1.set_duty_cycle(2,duty)
                            self.fan1.set_duty_cycle(3,duty)
                            self.fan2.set_duty_cycle(1,duty)
                            self.fan2.set_duty_cycle(2,duty)
                            self.fan2.set_duty_cycle(3,duty)
                            response=f'All fan duty set to {duty}'
                        except Exception as e:
                            response=f"Error setting duty cycle: {str(e)}"
                    
                        client_socket.send(response.encode('utf-8'))

                    elif data.startswith('set fan frequency:'):
                        try:
                            frequency= float(data.split(':')[1])
                            if 1<= frequency <=30:
                                self.fan1.set_frequency(frequency)
                                response = f"fan Frequency set to {frequency}"
                            else:
                                response = "Frequency must be between 1 and 30 kHZ"
                        except ValueError:
                            response = "Invalid frequency value"
                        except Exception as e:
                            response = f"Error setting frequency: {str(e)}"
                        client_socket.send(response.encode('utf-8'))


                        
                    elif data.startswith('set pump duty:'):
                        try:
                            duty = float(data.split(':')[1])
                            if 0 <= duty <= 100:
                                self.pump.set_duty_cycle(duty)
                                response = f"pump Duty cycle set to {duty}"
                            else:
                                response = "Duty cycle must be between 0 and 100"
                        except ValueError:
                            response = "Invalid duty cycle value"
                        except Exception as e:
                            response = f"Error setting duty cycle: {str(e)}"
                        client_socket.send(response.encode('utf-8'))

                    elif data.startswith('set pump frequency:'):
                        try:
                            frequency= float(data.split(':')[1])
                            if 1<= frequency <=30:
                                self.pump.set_frequency(frequency)
                                response = f"pump Frequency set to {frequency}"
                            else:
                                response = "Frequency must be between 1 and 30 kHZ"
                        except ValueError:
                            response = "Invalid frequency value"
                        except Exception as e:
                            response = f"Error setting frequency: {str(e)}"
                        client_socket.send(response.encode('utf-8'))
                    
                    elif data== 'open pump port':
                        try:
                            self.pump.turn_on
                            response= f'pump controller is opened'
                        except Exception as e:
                            response=f'Error open pump controller: {str(e)}'
                        client_socket.send(response.encode('utf-8'))
                    elif data== 'close pump port':
                        try:
                            self.pump.turn_off
                            response= f'pump controller is closed'
                        except Exception as e:
                            response=f'Error close pump controller: {str(e)}'
                        client_socket.send(response.encode('utf-8'))


                    else:
                        response = "Unknown command"
                        client_socket.send(response.encode('utf-8'))
                except ConnectionResetError:
                    print(f'Connection reset by client{client_address}')
                    break
                except Exception as e:
                    print(f"Error handling client{client_address}: {e}")
        
        finally:
            try:
                client_socket.close()
            except Exception as e:
                pass
            print(f"Connection closed with {client_address}")
    def start(self):
        self.adam_activate()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(5)
        server.settimeout(self.socket_timeout)
        print(f"Server listening on {self.host}:{self.port}")

        # 啟動溫度更新線程
        threading.Thread(target=self.update_temperatures, daemon=True).start()

        try:
            while self.running:
                try:
                    client_sock, address = server.accept()
                    print(f"Accepted connection from {address}")
                    client_handler = threading.Thread(target=self.handle_client, args=(client_sock,))
                    client_handler.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f'Error accepting connection:{e}')
                    if not self.running:
                        break
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            self.stop()
            server.close()

    def stop(self):
        self.running = False
        self.adam.stop_threading('buffer')
        self.adam.stop_threading('adam')
        self.adam.closeport()
        print('Experiment ended.')

if __name__ == "__main__":
    #開始實驗前記得查看網路IP位置，兩邊都需要查看
    server = CDUServer('192.168.227.57')
    try:
        server.start()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received in main thread")
    finally:
        print("Main thread exiting.")
