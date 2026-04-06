import socket
import json
from base_ctrl import BaseController
import time

def is_raspberry_pi5():
    try:
        with open('/proc/cpuinfo', 'r') as file:
            for line in file:
                if 'Model' in line:
                    return 'Raspberry Pi 5' in line
    except:
        return False
    return False

port = '/dev/ttyAMA0' if is_raspberry_pi5() else '/dev/serial0'
print(f"Ініціалізація бази на порту {port}...")

try:
    base = BaseController(port, 115200)
    time.sleep(1)
except Exception as e:
    print(f"Помилка підключення до бази: {e}")
    exit()

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"UDP Сервер готовий! Чекаю команди м/с на порту {UDP_PORT}...")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        command = json.loads(data.decode('utf-8'))

        print(f"Отримано від {addr}: {command}")
        
        l_speed = command.get("L", 0.0)
        r_speed = command.get("R", 0.0)
        
        if abs(l_speed) > 0.05 or abs(r_speed) > 0.05:
            base.send_command({"T": 1.0, "L": l_speed, "R": r_speed})
        else:
            base.send_command({"T": 0, "L": 0, "R": 0})
            
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nЗупинка...")
    base.send_command({"T": 0, "L": 0, "R": 0})
