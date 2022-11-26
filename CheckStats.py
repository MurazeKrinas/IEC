import threading
import signal
import array
import time
import psutil

def get_info(gpuStats, cpuStats, memStats):
    while True:
        with open('/sys/devices/gpu.0/load', 'r') as gpuInfo:
            gpuLoad = float(gpuInfo.read())/10

        with open('/proc/meminfo') as MemInfor:
            MemUsed = 0.0
            for line in MemInfor:
                s = line.split()
                if (s[0] == 'MemTotal:'):
                    MemUsed = int(s[1])
                elif (s[0] in ('MemFree:','Buffers:','Cached:')):
                    MemUsed -= int(s[1])
        MemUsed /= 1048576

        cpuLoad = psutil.cpu_percent()

        if gpuLoad != 0:
            gpuStats.append(gpuLoad)
        if cpuLoad != 0:
            cpuStats.append(cpuLoad)
        memStats.append(MemUsed)

        print(f'GPU: {gpuLoad}%,\tCPU: {cpuLoad}%,\tRAM: {MemUsed:.3f}/4GB')

        time.sleep(0.5)

if __name__ == '__main__':
    gpuStats = []
    cpuStats = []
    memStats = []

    t = threading.Thread(target=get_info, args=(gpuStats, cpuStats, memStats))
    t.daemon = True
    t.start()

    try:
        time.sleep(500)
    except KeyboardInterrupt:
        avgGpu = sum(gpuStats) / len(gpuStats)
        avgCpu = sum(cpuStats) / len(cpuStats)
        avgMem = sum(memStats) / len(memStats)

        with open('./Benmark.txt', 'a') as Benmark:
            Benmark.write(f'Avg GPU: {avgGpu:.3f}%,\tAvg CPU: {avgCpu:.3f}%,\tAvg Mem: {avgMem:.3f}/4GB\n')
        print(f'Avg GPU: {avgGpu:.3f}%,\tAvg CPU: {avgCpu:.3f}%,\tAvg Mem: {avgMem:.3f}/4GB\n')
        #print(f'GPU stats: {gpuStats}')
        #print(f'CPU stats: {cpuStats}')
