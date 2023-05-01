import serial
import time
from collections import *


arduino = serial.Serial(port='/dev/cu.SLAB_USBtoUART', baudrate=460800, timeout=.1)

number_of_samples = 100
def write(x):
    y = str(x)
    arduino.write(bytes(y, 'utf-8'))

start_time = time.time()
for i in range(number_of_samples):
    write(i)
    a = str(arduino.readline())
    print(a)

end_time = time.time()
latency = (end_time - start_time)/2/number_of_samples*1000
print('latency: ', latency)
