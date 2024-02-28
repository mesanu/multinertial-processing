# multinertial-processing
Visualization suite for the multinertial project, built on Ursina

Running the live visualization suite requires assembled hardware, however
a visualization of the basic gradient descent algorithm can be run locally
using gradient_descent_vis.py

## Running gradient_descent_vis.py
Ensure the modules in requirements.txt are installed

Quickstart command
>python gradient_descent_vis.py -amax 15 -amin 5 -lr 0.01 -n 90

Run gradient_descent_vis.py -h for a description of these arguments. The values
above are good starting points that display nice results.

This script generates 3 sets of basis vectors, then rotates each by a random amount
between amin and amax on a random axis of rotation. The blue cubes are the tips of 3 sets
of basis vectors to fit to. The red cubes are the attempted best fit (guaged by euclidean distance) using gradient descent.