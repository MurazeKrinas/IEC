import threading
import signal
import array
import time
import psutil

def get_info(gpuStats, cpuStats):
    while True:
        with open('/sys/devices/gpu.0/load', 'r') as gpuInfo:
            gpuLoad = float(gpuInfo.read())/10

        cpuLoad = psutil.cpu_percent()

        if gpuLoad != 0:
            gpuStats.append(gpuLoad)
        if cpuLoad != 0:
            cpuStats.append(cpuLoad)

        print(f'GPU: {gpuLoad}%,\tCPU: {cpuLoad}%')

        time.sleep(1.5)

if __name__ == '__main__':
    gpuStats = []
    cpuStats = []

    t = threading.Thread(target=get_info, args=(gpuStats, cpuStats))
    t.daemon = True
    t.start()

    try:
        time.sleep(1200)
    except KeyboardInterrupt:
        avgGpu = sum(gpuStats) / len(gpuStats)
        avgCpu = sum(cpuStats) / len(cpuStats)
        
        with open('./IEC/Benmark.txt', 'a') as Benmark:
            Benmark.write(f'Avg GPU: {avgGpu:.3f}%,\tAvg CPU: {avgCpu:.3f}%\n')
        #print(f'GPU stats: {gpuStats}')
        #print(f'CPU stats: {cpuStats}')
