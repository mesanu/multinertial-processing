import numpy as np
from math import sin, cos, radians

def rotateVectorsAboutAxis(vectors, axis, degrees):
        theta = radians(degrees)
        sinTheta = sin(theta)
        cosTheta = cos(theta)
        oneMinusCos = 1 - cosTheta
        mulXY = axis[0] * axis[1]
        mulYZ = axis[1] * axis[2]
        mulXZ = axis[0] * axis[2]

        rotMat = np.array([[cosTheta + axis[0]**2 * oneMinusCos,
                            mulXY * oneMinusCos - axis[2] * sinTheta,
                            mulXZ * oneMinusCos + axis[1] * sinTheta],
                           [mulXY * oneMinusCos + axis[2] * sinTheta,
                            cosTheta + axis[1]**2 * oneMinusCos,
                            mulYZ * oneMinusCos - axis[0] * sinTheta],
                           [mulXZ * oneMinusCos - axis[1] * sinTheta,
                            mulYZ * oneMinusCos + axis[0] * sinTheta,
                            cosTheta + axis[2]**2 * oneMinusCos]])
        return np.matmul(rotMat, vectors.T).T

def getDataBlock(conn, gyroDPS, accelGRange):
    i = 0
    data = conn.recv(4096)
    index = data[i]
    i += 1
    headSecondsTs = float(int.from_bytes(data[i:(i+8)], byteorder='little'))/10**6
    i += 8
    tailSecondsTs = float(int.from_bytes(data[i:(i+8)], byteorder='little'))/10**6
    i += 8
    numFrames = int.from_bytes(data[i:(i+2)], byteorder='little')
    i += 2

    gyroX = data[i:i+numFrames*2]
    gyroX = [float(int.from_bytes(gyroX[n:n+2], byteorder='little', signed=True))/32768*gyroDPS for n in range(0, numFrames*2, 2)]
    i += numFrames*2

    gyroY = data[i:i+numFrames*2]
    gyroY = [float(int.from_bytes(gyroY[n:n+2], byteorder='little', signed=True))/32768*gyroDPS for n in range(0, numFrames*2, 2)]
    i += numFrames*2

    gyroZ = data[i:i+numFrames*2]
    gyroZ = [float(int.from_bytes(gyroZ[n:n+2], byteorder='little', signed=True))/32768*gyroDPS for n in range(0, numFrames*2, 2)]
    i += numFrames*2

    accelX = data[i:i+numFrames*2]
    accelX = [float(int.from_bytes(accelX[n:n+2], byteorder='little', signed=True))/32768*accelGRange for n in range(0, numFrames*2, 2)]
    accelX = np.array(accelX)
    i += numFrames*2

    accelY = data[i:i+numFrames*2]
    accelY = [float(int.from_bytes(accelY[n:n+2], byteorder='little', signed=True))/32768*accelGRange for n in range(0, numFrames*2, 2)]
    accelY = np.array(accelY)
    i += numFrames*2

    accelZ = data[i:i+numFrames*2]
    accelZ = [float(int.from_bytes(accelZ[n:n+2], byteorder='little', signed=True))/32768*accelGRange for n in range(0, numFrames*2, 2)]
    accelZ = np.array(accelZ)
    i += numFrames*2

    timestamps = np.linspace(tailSecondsTs, headSecondsTs, numFrames+1)[1:]

    return index, np.column_stack([timestamps, gyroX, gyroY, gyroZ, accelX, accelY, accelZ])

def sendStartCommand(conn):
    conn.send(b"start\0")

def sendStopCommand(conn):
    conn.send(b"stop\0")

def sendCalCommand(conn):
    conn.send(b"cal\0")

def sendAck(conn):
    conn.send(b"ack\0")
