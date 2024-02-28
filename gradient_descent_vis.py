from gradient_descent import *

from ursina import *
import time
import argparse

class BasisVectorVisEntities:
    def __init__(self, color):
        self.centerEntity = Entity(position=(0, 0, 0))
        self.basisXEntity = Entity(position=(1, 0, 0), color=color, model='cube', scale=0.1)
        self.basisYEntity = Entity(position=(0, -1, 0), color=color, model='cube', scale=0.1)
        self.basisZEntity = Entity(position=(0, 0, 1), color=color, model='cube', scale=0.1)

        self.basisXEntity.look_at(self.centerEntity, axis='forward')
        self.basisYEntity.look_at(self.centerEntity, axis='forward')
        self.basisZEntity.look_at(self.centerEntity, axis='forward')

    def updateEntities(self, basisVectors):
        yFlip = np.array([1, -1, 1])

        self.basisXEntity.position = basisVectors[0]
        self.basisYEntity.position = basisVectors[1]
        self.basisZEntity.position = basisVectors[2]

        self.basisXEntity.look_at(self.centerEntity, axis='forward')
        self.basisYEntity.look_at(self.centerEntity, axis='forward')
        self.basisZEntity.look_at(self.centerEntity, axis='forward')

class InputWrap(Entity):
    def __init__(self, camera, **kwargs):
        super().__init__()
        self.camera = camera
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

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-amax', help='maximum random rotation angle', type=float)
    argParser.add_argument('-amin', help='minimum random rotation angle', type=float)
    argParser.add_argument('-lr', help='learning rate', type=float)
    argParser.add_argument('-n', help='number of gradient descent iterations', type=int)
    argParser.add_argument('-v', help='verbose mode, print out the error between iterations',
                                action=argparse.BooleanOptionalAction)

    args = argParser.parse_args()

    basisVectors = np.array([np.eye(3), np.eye(3), np.eye(3)])

    minmaxDiff = args.amax - args.amin

    for i in range(len(basisVectors)):
        randomRotationDegrees = random.random() * minmaxDiff + args.amin
        randomRotationAxis = (np.random.rand(3)) - 0.5
        randomRotationAxis /= np.linalg.norm(randomRotationAxis)
        basisVectors[i] = rotateVectorsAboutAxis(basisVectors[i].T, randomRotationAxis, randomRotationDegrees).T

    refBasis = np.copy(basisVectors[0])

    if(args.v):
        prevLeastSquares = refBasis
        for i in range(args.n):
            leastSquares, updated = getLeastSquaresOrientation(prevLeastSquares, basisVectors, 1, args.lr)
            error = getTotalAverageError(leastSquares, basisVectors)
            print("{}, {:.4f}".format(i, error))
            prevLeastSquares = np.copy(leastSquares)
    else:
        startTime = time.time_ns()
        leastSquares, updated = getLeastSquaresOrientation(refBasis, basisVectors, args.n, args.lr)
        endTime = time.time_ns()

    app = Ursina(borderless = False)
    visEntities = []

    for vectorSet in basisVectors:
        visEntities = BasisVectorVisEntities(color.blue)
        visEntities.updateEntities(vectorSet)

    refEntity = BasisVectorVisEntities(color.red)
    refEntity.updateEntities(leastSquares)

    camera.look_at(refEntity.centerEntity)

    wrapper = InputWrap(camera)

    if(not args.v):
        print("Time to run: {} us".format((endTime - startTime)/1000))

    app.run()

if __name__ == '__main__':
    sys.exit(main())