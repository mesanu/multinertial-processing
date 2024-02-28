import numpy as np
from math import degrees, sin, cos
from common import *

def getTotalAverageError(refPoints, targetPoints):
    xDiffs = (refPoints[:,0] - targetPoints[:,:,0])
    yDiffs = (refPoints[:,1] - targetPoints[:,:,1])
    zDiffs = (refPoints[:,2] - targetPoints[:,:,2])

    totalError = 0

    for i in range(len(targetPoints)):
        totalError += np.mean(np.linalg.norm(np.column_stack([xDiffs[i], yDiffs[i], zDiffs[i]]), axis=1))
    
    return totalError


def getGradThetaMatrix(axis):
    mulXY = axis[0] * axis[1]
    mulYZ = axis[1] * axis[2]
    mulXZ = axis[0] * axis[2]

    mat = np.array([[axis[0]**2, mulXY - axis[2], mulXZ + axis[1]],
                    [mulXY + axis[2], axis[1]**2, mulYZ - axis[0]],
                    [mulXZ - axis[1], mulYZ + axis[0], axis[2]**2]])
    return mat

def getGradientTheta(refPoints, targetPoints, axis):
    xDiffs = (refPoints[:,0] - targetPoints[:,0])
    yDiffs = (refPoints[:,1] - targetPoints[:,1])
    zDiffs = (refPoints[:,2] - targetPoints[:,2])

    totalDiffs = np.column_stack([xDiffs, yDiffs, zDiffs])

    diffNorms = np.linalg.norm(totalDiffs, axis=1)

    if(np.any(diffNorms == 0)):
        return 0

    gradThetaMat = getGradThetaMatrix(axis)

    thetaGrads = np.sum(totalDiffs*np.matmul(gradThetaMat, refPoints.T).T, axis=1)/diffNorms

    return np.array(np.mean(thetaGrads))


def getLeastSquaresOrientation(refBasis, targetPoints, numIterations, learningRate):
    i = numIterations
    localRefBasis = np.copy(refBasis)
    prevRefBasis = np.copy(refBasis)

    dims = np.arange(3)

    updated = False

    while(i):
        gradient = np.zeros(3)
        for dim in dims:
            for pointSet in targetPoints:
                gradient[dim] += getGradientTheta(localRefBasis, pointSet, localRefBasis[dim])
        gradientNorm = np.linalg.norm(gradient)
        gradient /= gradientNorm
        axisOfRotation = np.matmul(localRefBasis.T, gradient)
        localRefBasis = rotateVectorsAboutAxis(localRefBasis, axisOfRotation, degrees(gradientNorm)*-1*learningRate)

        refBasisError = getTotalAverageError(localRefBasis, targetPoints)
        
        prevRefBasisError = getTotalAverageError(prevRefBasis, targetPoints)

        if(prevRefBasisError < refBasisError):
            learningRate *= 0.5
            localRefBasis = np.copy(prevRefBasis)
            updated = True
        else:
            prevRefBasis = np.copy(localRefBasis)
            updated = True

        i -= 1
    return localRefBasis, updated