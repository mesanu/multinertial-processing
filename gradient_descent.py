import numpy as np
from math import degrees
from common import *

def get_total_error(ref_points, target_points):
    """ Return the total mean squared error between ref_points and target_points """
    x_diffs = (ref_points[:,0] - target_points[:,:,0])
    y_diffs = (ref_points[:,1] - target_points[:,:,1])
    z_diffs = (ref_points[:,2] - target_points[:,:,2])

    total_error = 0

    for i in range(len(target_points)):
        total_error += np.mean(
            np.linalg.norm(np.column_stack([x_diffs[i], y_diffs[i], z_diffs[i]]),
                           axis=1)
            )

    return total_error


def get_grad_theta_matrix(axis):
    """ Return the matrix used to derive the gradient along the theta direction """
    mul_xy = axis[0] * axis[1]
    mul_yz = axis[1] * axis[2]
    mul_xz = axis[0] * axis[2]

    mat = np.array([[axis[0]**2, mul_xy - axis[2], mul_xz + axis[1]],
                    [mul_xy + axis[2], axis[1]**2, mul_yz - axis[0]],
                    [mul_xz - axis[1], mul_yz + axis[0], axis[2]**2]])
    return mat

def get_gradient_theta(ref_points, target_points, axis):
    """ Return the gradient along the theta direction """
    x_diffs = (ref_points[:,0] - target_points[:,0])
    y_diffs = (ref_points[:,1] - target_points[:,1])
    z_diffs = (ref_points[:,2] - target_points[:,2])

    total_diffs = np.column_stack([x_diffs, y_diffs, z_diffs])

    diff_norms = np.linalg.norm(total_diffs, axis=1)

    if np.any(diff_norms == 0):
        return 0

    grad_theta_mat = get_grad_theta_matrix(axis)

    theta_grads = np.sum(total_diffs*np.matmul(grad_theta_mat, ref_points.T).T, axis=1)/diff_norms

    return np.array(np.mean(theta_grads))


def get_least_squares_orientation(ref_basis, target_points, num_iterations, learning_rate):
    """ Return a set of basis vectors with minimum distance to vectors in target_points """
    i = num_iterations
    local_ref_basis = np.copy(ref_basis)
    prev_ref_basis = np.copy(ref_basis)

    dims = np.arange(3)

    updated = False

    while i:
        gradient = np.zeros(3)
        for dim in dims:
            for point_set in target_points:
                gradient[dim] += get_gradient_theta(
                    local_ref_basis,
                    point_set,
                    local_ref_basis[dim]
                )
        gradient_norm = np.linalg.norm(gradient)
        gradient /= gradient_norm
        axis_of_rotation = np.matmul(local_ref_basis.T, gradient)

        local_ref_basis = rotate_vectors_about_axis(
            local_ref_basis, axis_of_rotation,
            degrees(gradient_norm)*-1*learning_rate
        )

        ref_basis_error = get_total_error(local_ref_basis, target_points)

        prev_ref_basis_error = get_total_error(prev_ref_basis, target_points)

        if prev_ref_basis_error < ref_basis_error:
            learning_rate *= 0.5
            local_ref_basis = np.copy(prev_ref_basis)
            updated = True
        else:
            prev_ref_basis = np.copy(local_ref_basis)
            updated = True

        i -= 1
    return local_ref_basis, updated
