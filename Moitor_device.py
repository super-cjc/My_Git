import threading
import time
import logging
import serial



#配置文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class DeviceMannger(object):
    def __init__(self,port):
        #连接状态
        self.voice = False
        self.locked = threading.Lock()
        self.start_time = 0
        self.last_time = 0
        self.port = port
        self.msg = 0

        self.ser =  serial.Serial(
            port=port,  # 串口名称，Linux系统通常为/dev/ttyUSB0, /dev/ttyUSB1等
            baudrate=9600,  # 波特率
            parity=serial.PARITY_NONE,  # 校验位
            stopbits=serial.STOPBITS_ONE,  # 停止位
            bytesize=serial.EIGHTBITS,  # 数据位
            timeout=1  # 读取超时时间，单位为秒
    )


    def Check_Serial(self):
        """检查串口可以用的接口"""
        status = False
        import serial.tools.list_ports
        serials_aviable = serial.tools.list_ports.comports()
        for serial in serials_aviable:
            if serial.device == self.port:
                status = True
                break
        return status

    def Connect_Serial(self):
        """连接串口打印数据"""
        if self.ser.is_open:
            while self.ser.in_waiting>0:
                self.msg = self.ser.readline()
                print(self.msg)


    def Disconnect_serial(self):
        """断开串口连接"""
        self.ser.close()
        pass

    def Moitor_Device(self):
        """实时监控连接设备"""
        while True:
            with self.locked:
                is_connect = False
                is_connect = self.Check_Serial()
                if is_connect and self.voice == False:
                    try:
                        self.voice = True
                        print("语音设备已连接")
                        self.start_time = time.time()
                    except:
                        self.voice = False
                        print("连接失败")
                    finally:
                        self.Disconnect_serial()
                elif not is_connect:
                    self.voice = False
                    self.last_time = time.time()
                    print("当前设备未进行连接，等待连接中")
            time.sleep(1)





if __name__ == "__main__":
    DEVICE = DeviceMannger('COM4')
    check_device = threading.Thread(target=DEVICE.Moitor_Device,daemon=True).start()
    a = 0
    while True:
        a = a + 1
        time.sleep(1)
        # print(a)

