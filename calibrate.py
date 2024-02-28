import socket
import numpy as np
import yaml
from common import *
import time

NUM_ORIENTATIONS = 6

calFile = open("./calibrations.yaml", "r")
calibrations = yaml.safe_load(calFile)
calFile.close()

configs = yaml.safe_load(open("./config.yaml", "r"))

host = configs["connection"]["host"]
port = int(configs["connection"]["port"])

numFrames = int(configs["imu"]["calFrameCount"])
gyroDPSRange = float(configs["imu"]["gyro"]["dpsRange"])
accelGRange = float(configs["imu"]["accel"]["gRange"])
numImus = int(configs["imu"]["numImus"])

print("connecting")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))

sendCalCommand(s)

time.sleep(1)

calDicts = []

for i in range(NUM_ORIENTATIONS):
    calDictsOrientation = []
    frameMeans = [[] for _ in range(numImus)]
    frameCounts = np.array([numFrames]*numImus)
    sendStartCommand(s)
    while(np.any(frameCounts > 0)):
        index, imuFrame = getDataBlock(s, gyroDPSRange, accelGRange)
        #print(imuFrame)
        if(frameCounts[index] > 0):
            frameMeans[index].append(np.mean(imuFrame[:,1:], axis=0))
            frameCounts[index] -= 1
        if(np.any(frameCounts > 0)):
            sendAck(s)
        else:
            sendStopCommand(s)

    frameMeans = np.mean(np.array(frameMeans), axis=1)
    for mean in frameMeans:
        calDict = dict({"gyro":{"x":0, "y":0, "z":0}, "accel":{"x":0, "y":0, "z":0}})
        calDict["gyro"]["x"] = float(mean[0])
        calDict["gyro"]["y"] = float(mean[1])
        calDict["gyro"]["z"] = float(mean[2])
        calDict["accel"]["x"] = float(mean[3])
        calDict["accel"]["y"] = float(mean[4])
        calDict["accel"]["z"] = float(mean[5])
        calDictsOrientation.append(calDict)

    calDicts.append(calDictsOrientation)
    print("Finished orientation")

sendCalCommand(s)

configFile = open("./calibrations.yaml", "w")
configFile.write(yaml.dump(calDicts))
configFile.close()