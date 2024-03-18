from ursina import *
from gradient_descent import *

import time
import argparse

class BasisVectorVisEntities:
    """ Class for containing set of entities used to visualize basis vectors """
    def __init__(self, cube_colour):
        self.center_entity = Entity(position=(0, 0, 0))
        self.basis_x_entity = Entity(position=(1, 0, 0), color=cube_colour, model='cube', scale=0.1)
        self.basis_y_entity = Entity(position=(0, -1, 0), color=cube_colour, model='cube', scale=0.1)
        self.basis_z_entity = Entity(position=(0, 0, 1), color=cube_colour, model='cube', scale=0.1)

        self.basis_x_entity.look_at(self.center_entity, axis='forward')
        self.basis_y_entity.look_at(self.center_entity, axis='forward')
        self.basis_z_entity.look_at(self.center_entity, axis='forward')

    def update_entities(self, basis_vectors):
        """ Update visualization entities """
        self.basis_x_entity.position = basis_vectors[0]
        self.basis_y_entity.position = basis_vectors[1]
        self.basis_z_entity.position = basis_vectors[2]

        self.basis_x_entity.look_at(self.center_entity, axis='forward')
        self.basis_y_entity.look_at(self.center_entity, axis='forward')
        self.basis_z_entity.look_at(self.center_entity, axis='forward')

class InputWrap(Entity):
    """ Wrapper class to allow for camera control without redefining input() """
    def __init__(self, camera_entity, **kwargs):
        super().__init__()
        self.camera_entity = camera_entity
        for key, value in kwargs.items():
            setattr(self, key, value)

    def input(self, key):
        """ Definition for input handling for Ursina entities """
        if(key == 'e hold' or key == 'e'):
            self.camera_entity.position += self.camera_entity.up*0.5
        elif(key == 'q hold' or key == 'q'):
            self.camera_entity.position += self.camera_entity.down*0.5
        elif(key == 'a hold' or key == 'a'):
            self.camera_entity.position += self.camera_entity.left*0.5
        elif(key == 'd hold' or key == 'd'):
            self.camera_entity.position += self.camera_entity.right*0.5
        elif(key == 'w hold' or key == 'w'):
            self.camera_entity.position += self.camera_entity.forward*0.5
        elif(key == 's hold' or key == 's'):
            self.camera_entity.position += self.camera_entity.back*0.5
        elif(key == 'left arrow hold' or key == 'left arrow'):
            self.camera_entity.rotate((0, -1, 0))
        elif(key == 'right arrow hold' or key == 'right arrow'):
            self.camera_entity.rotate((0, 1, 0))
        elif(key == 'up arrow hold' or key == 'up arrow'):
            self.camera_entity.rotate((-1, 0, 0))
        elif(key == 'down arrow hold' or key == 'down arrow'):
            self.camera_entity.rotate((1, 0, 0))

def main():
    """ Main function for visualization """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-amax', help='maximum random rotation angle', type=float)
    arg_parser.add_argument('-amin', help='minimum random rotation angle', type=float)
    arg_parser.add_argument('-lr', help='learning rate', type=float)
    arg_parser.add_argument('-n', help='number of gradient descent iterations', type=int)
    arg_parser.add_argument('-v', help='verbose mode, print out the error between iterations',
                                action=argparse.BooleanOptionalAction)

    args = arg_parser.parse_args()

    basis_vectors = np.array([np.eye(3), np.eye(3), np.eye(3)])

    minmax_diff = args.amax - args.amin

    for i in range(len(basis_vectors)):
        random_rotation_degrees = random.random() * minmax_diff + args.amin
        random_rotation_axis = (np.random.rand(3)) - 0.5
        random_rotation_axis /= np.linalg.norm(random_rotation_axis)
        basis_vectors[i] = rotate_vectors_about_axis(basis_vectors[i].T, random_rotation_axis, random_rotation_degrees).T

    ref_basis = np.copy(basis_vectors[0])

    if(args.v):
        prev_least_squares = ref_basis
        for i in range(args.n):
            least_squares, _ = get_least_squares_orientation(prev_least_squares, basis_vectors, 1, args.lr)
            error = get_total_error(least_squares, basis_vectors)
            print("{}, {:.4f}".format(i, error))
            prev_least_squares = np.copy(least_squares)
    else:
        start_time = time.time_ns()
        least_squares, _ = get_least_squares_orientation(ref_basis, basis_vectors, args.n, args.lr)
        end_time = time.time_ns()

    ursina_app = Ursina(borderless = False)
    vis_entities = []

    for vector_set in basis_vectors:
        vis_entities = BasisVectorVisEntities(color.blue)
        vis_entities.update_entities(vector_set)

    ref_entity = BasisVectorVisEntities(color.red)
    ref_entity.update_entities(least_squares)

    camera.look_at(ref_entity.center_entity)

    wrapper = InputWrap(camera)

    if not args.v:
        print("Time to run: {} us".format((end_time - start_time)/1000))

    ursina_app.run()

if __name__ == '__main__':
    sys.exit(main())
