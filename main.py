import socket
import numpy as np
from multiprocessing import Process, Queue, Event
from math import floor
from common import *
from gradient_descent import *
import yaml
from ursina import *
import time

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

AVG_INDEX = 0
LEAST_SQUARES_INDEX = 1

class InputWrap(Entity):
    def __init__(self, camera, resetEvents, **kwargs):
        super().__init__()
        self.camera = camera
        self.resetEvents = resetEvents
        for key, value in kwargs.items():
            setattr(self, key, value)

    def input(self, key):
        if(key == 'e hold' or key == 'e'):
            self.camera.position += self.camera.up*0.5
        elif(key == 'q hold' or key == 'q'):
            self.camera.position += self.camera.down*0.5
        elif(key == 'a hold' or key == 'a'):
            self.camera.position += self.camera.left*0.5
        elif(key == 'd hold' or key == 'd'):
            self.camera.position += self.camera.right*0.5
        elif(key == 'w hold' or key == 'w'):
            self.camera.position += self.camera.forward*0.5
        elif(key == 's hold' or key == 's'):
            self.camera.position += self.camera.back*0.5
        elif(key == 'left arrow hold' or key == 'left arrow'):
            self.camera.rotate((0, -1, 0))
        elif(key == 'right arrow hold' or key == 'right arrow'):
            self.camera.rotate((0, 1, 0))
        elif(key == 'up arrow hold' or key == 'up arrow'):
            self.camera.rotate((-1, 0, 0))
        elif(key == 'down arrow hold' or key == 'down arrow'):
            self.camera.rotate((1, 0, 0))
        elif(key == 'space'):
            for event in self.resetEvents:
                event.set()

class AccumulatorEntities:
    def __init__(self, centerPosition, texture):
        self.axisFlip = np.array([-1, -1, 1])
        self.centerPosition = centerPosition * self.axisFlip
        self.centerEntity = Entity(model='cube', texture=texture, scale=1, position=self.centerPosition )
        self.basisXEntity = Entity(position=((-1, 0, 0) + self.centerPosition ))
        self.basisYEntity = Entity(position=((0, -1, 0) + self.centerPosition ))
        self.basisZEntity = Entity(position=(0, 0, 1) + self.centerPosition )

        self.centerEntity.look_at(self.basisXEntity, axis='right')
        self.centerEntity.look_at(self.basisYEntity, axis='up')
        self.centerEntity.look_at(self.basisZEntity, axis='forward')
    
    def updateEntities(self, basisVectors):
        self.basisXEntity.position = (basisVectors[0,:] * self.axisFlip + self.centerPosition)
        self.basisYEntity.position = (basisVectors[1,:] * self.axisFlip + self.centerPosition)
        self.basisZEntity.position = (basisVectors[2,:] * self.axisFlip + self.centerPosition)

        self.centerEntity.look_at(self.basisXEntity, axis='right')
        self.centerEntity.look_at(self.basisYEntity, axis='up')
        self.centerEntity.look_at(self.basisZEntity, axis='forward')


class Accumulator:
    def __init__(self, numImus, renderQ, resetEvent, renderPeriod, idealSamplePeriod, index):
        self.lastRawDataRow = [None for _ in range(numImus)]
        self.interpolationLeftovers = [[] for _ in range(numImus)]
        self.nextRenderTs =  None

        self.idealSamplePeriod = idealSamplePeriod

        self.index = index

        self.interpolationStartTs = None

        self.normAxes = np.eye(3)

        self.renderQ = renderQ
        self.resetEvent = resetEvent
        self.renderPeriod = renderPeriod

        self.gyroCals = np.zeros((numImus, 3))

        self.frameBuffers = [[] for _ in range(numImus)]

        self.numImus = numImus

    def _interpolateFrames(self, combinedFrames):
        if(self.interpolationStartTs is None):
            self.interpolationStartTs = np.max(combinedFrames[:,0,0])
        leastLastFrameTs = np.min(combinedFrames[:,-1, 0])

        numInterpolationPeriods = floor((leastLastFrameTs - self.interpolationStartTs)/self.idealSamplePeriod)
        interpolationEndTs = (self.idealSamplePeriod * numInterpolationPeriods) + self.interpolationStartTs

        interpolatedXPoints = np.linspace(self.interpolationStartTs, interpolationEndTs, numInterpolationPeriods + 1)

        interpolatedFrames = np.zeros([len(combinedFrames), len(interpolatedXPoints), 7])

        interpolatedFrames[:,:,0] = interpolatedXPoints

        for i, frame in enumerate(combinedFrames):
            if(len(self.interpolationLeftovers[i]) > 0):
                totalFrame = np.vstack([self.interpolationLeftovers[i].pop(0), frame])
            else:
                totalFrame = frame
            for j in range(1, totalFrame.shape[1]):
                interpolatedFrames[i,:,j] = np.interp(interpolatedXPoints, totalFrame[:,0], totalFrame[:,j])
            leftovers = totalFrame[totalFrame[:,0] > interpolationEndTs]
            if(len(leftovers) > 0):
                self.interpolationLeftovers[i].append(leftovers)

        self.interpolationStartTs = interpolationEndTs + self.idealSamplePeriod
        return interpolatedFrames

    def integratePositionOrientation(self, imuIndex, frame):
        if(self.resetEvent[self.index].is_set()):
            self._reset()
        if(self.nextRenderTs is None):
            self.nextRenderTs = frame[0,0] + self.renderPeriod
        if(self.lastRawDataRow[imuIndex] is None):
            self.lastRawDataRow[imuIndex] = np.copy(frame[-1])
        else:
            frame = np.vstack([self.lastRawDataRow[imuIndex], frame])
            self.lastRawDataRow[imuIndex] = np.copy(frame[-1])

        calFrame = frame[:-1]

        calFrame[:,1] -= self.gyroCals[imuIndex,0]
        calFrame[:,2] -= self.gyroCals[imuIndex,1]
        calFrame[:,3] -= self.gyroCals[imuIndex,2]

        self.frameBuffers[imuIndex].append(calFrame)

        for queue in self.frameBuffers:
            if(len(queue) == 0):
                return

        nextFrameShape = self.frameBuffers[0][0].shape

        combinedFrames = np.zeros([self.numImus, nextFrameShape[0], nextFrameShape[1]])

        for i,queue in enumerate(self.frameBuffers):
            combinedFrames[i] = queue.pop(0)

        interpolatedFrames = self._interpolateFrames(combinedFrames)

        self._processInterpolatedFrame(interpolatedFrames)
    
    def setCalibrations(self, calOrientations):
        dimOrder = ["x", "y", "z"]
        for i, calOrientation in enumerate(calOrientations):
            dimIndex = int(i/2)
            dim = dimOrder[dimIndex]
            for j, imuCal in enumerate(calOrientation):
                if(j >= self.numImus):
                    break
                self.gyroCals[j] += np.array([float(imuCal["gyro"]["x"]), float(imuCal["gyro"]["y"]), float(imuCal["gyro"]["z"])])
        
        self.gyroCals /= len(calOrientations)

class AverageAccumulator(Accumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _processInterpolatedFrame(self, interpolatedFrames):
        avgFrame = np.mean(interpolatedFrames[:,:,1:], axis=0)
        gyroData = avgFrame[:,:3]

        ratesOfRotation = np.linalg.norm(gyroData, axis=1)
        axesOfRotation = gyroData/ratesOfRotation[:, None]

        rotationAngles = ratesOfRotation*self.idealSamplePeriod
        for i in range(len(ratesOfRotation)):
            self.normAxes = rotateVectorsAboutAxis(self.normAxes, np.matmul(self.normAxes.T, axesOfRotation[i]), rotationAngles[i])
            if(interpolatedFrames[0,i,0] > self.nextRenderTs):
                self.renderQ.put([self.index, np.copy(self.normAxes)])
                self.nextRenderTs += self.renderPeriod

    def _reset(self):
        self.normAxes = np.eye(3)
        self.resetEvent[self.index].clear()

class LeastSquaresAccumulator(Accumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imuNormAxes = np.array([np.eye(3), np.eye(3), np.eye(3)])

    def _processInterpolatedFrame(self, interpolatedFrames):
        gyroData = interpolatedFrames[:,:,1:4]

        ratesOfRotation = np.linalg.norm(gyroData, axis=2)
        axesOfRotation = gyroData/ratesOfRotation[:,:,None]

        rotationAngles = ratesOfRotation*self.idealSamplePeriod
        for i in range(ratesOfRotation.shape[1]):
            for j in range(self.numImus):
                self.imuNormAxes[j] = rotateVectorsAboutAxis(self.imuNormAxes[j], np.matmul(self.imuNormAxes[j].T, axesOfRotation[j,i]), rotationAngles[j,i])
            if(interpolatedFrames[0,i,0] > self.nextRenderTs):
                self.normAxes, updated = getLeastSquaresOrientation(self.imuNormAxes[1], self.imuNormAxes, 80, 0.1)
                if(updated):
                    self.imuNormAxes = np.array([self.normAxes] * self.numImus)
                self.renderQ.put([self.index, np.copy(self.normAxes)])
                self.nextRenderTs += self.renderPeriod

    def _reset(self):
        self.normAxes = np.eye(3)
        self.imuNormAxes = np.array([np.eye(3), np.eye(3), np.eye(3)])
        self.resetEvent[self.index].clear()

def visualize(renderQ, resetEvents,):
    app = Ursina(borderless = False)
    inputWrap = InputWrap(camera, resetEvents)
    accumulatorModels = [AccumulatorEntities((-2, 0, 0), 'brick'),
                         AccumulatorEntities((2, 0, 0), 'brick')]

    errorDisplayStrings = ["Average error from start: {:.4f}",
                           "Least squares error from start: {:.4f}"]

    errorDisplays = [Text(text=errorDisplayStrings[0].format(0), position=(-0.8, 0.3)),
                     Text(text=errorDisplayStrings[1].format(0), position=(-0.8, 0.2))]

    origin = Entity(position=(0,0,0))

    camera.position = (0, 25, 0)
    camera.look_at(origin)
    camera.rotate((0, 0, 180))
    while(1):
        data = renderQ.get()
        index = data[0]
        basisVectors = data[1]
        accumulatorModels[index].updateEntities(basisVectors)
        errorFromStart = np.mean(np.arccos(np.clip(np.sum(basisVectors*np.eye(3), axis=1),-1.0,1.0)))
        errorDisplays[index].text = errorDisplayStrings[index].format(errorFromStart)
        app.step()

def main():
    configs = yaml.safe_load(open("./config.yaml", "r"))

    host = configs["connection"]["host"]
    port = int(configs["connection"]["port"])
    accelGRange = float(configs["imu"]["accel"]["gRange"])
    gyroDPSRange = float(configs["imu"]["gyro"]["dpsRange"])

    numImus = int(configs["imu"]["numImus"])

    idealSamplePeriod = float(configs["imu"]["idealSamplePeriod"])
    visUpdatePeriod = float(configs["visualizer"]["renderPeriod"])

    calibrations = yaml.safe_load(open("./calibrations.yaml"))

    renderQ = Queue()
    resetEvents = [Event(), Event()]

    avgAccumulator = AverageAccumulator(numImus,
                                        renderQ,
                                        resetEvents,
                                        visUpdatePeriod,
                                        idealSamplePeriod,
                                        AVG_INDEX)
    avgAccumulator.setCalibrations(calibrations)

    leastSquaresAccumulator = LeastSquaresAccumulator(numImus,
                                                      renderQ,
                                                      resetEvents,
                                                      visUpdatePeriod,
                                                      idealSamplePeriod,
                                                      LEAST_SQUARES_INDEX)
    leastSquaresAccumulator.setCalibrations(calibrations)

    renderP = Process(target=visualize, args=(renderQ, resetEvents,))

    renderP.start()

    time.sleep(2)

    print("connecting")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
    sendStartCommand(s)
    while (1):
        imuIndex, imuFrame = getDataBlock(s, gyroDPSRange, accelGRange)
        avgAccumulator.integratePositionOrientation(imuIndex, imuFrame)
        leastSquaresAccumulator.integratePositionOrientation(imuIndex, imuFrame)
        sendAck(s)

if __name__ == '__main__':
    sys.exit(main())
