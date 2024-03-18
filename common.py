from math import sin, cos, radians
import numpy as np

def rotate_vectors_about_axis(vectors, axis, degrees):
    """ Rotate a set of vectors about a specified axis """
    theta = radians(degrees)
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    one_minus_cos = 1 - cos_theta
    mul_xy = axis[0] * axis[1]
    mul_yz = axis[1] * axis[2]
    mul_xz = axis[0] * axis[2]

    rot_mat = np.array([[cos_theta + axis[0]**2 * one_minus_cos,
                        mul_xy * one_minus_cos - axis[2] * sin_theta,
                        mul_xz * one_minus_cos + axis[1] * sin_theta],
                        [mul_xy * one_minus_cos + axis[2] * sin_theta,
                        cos_theta + axis[1]**2 * one_minus_cos,
                        mul_yz * one_minus_cos - axis[0] * sin_theta],
                        [mul_xz * one_minus_cos - axis[1] * sin_theta,
                        mul_yz * one_minus_cos + axis[0] * sin_theta,
                        cos_theta + axis[2]**2 * one_minus_cos]])
    return np.matmul(rot_mat, vectors.T).T

def get_data_block(conn, gyro_dps, accel_g_range):
    """ Extract a block of data from ESP32 on breadboard and parse the binary string"""
    i = 0
    data = conn.recv(4096)
    index = data[i]
    i += 1
    head_seconds_ts = float(int.from_bytes(data[i:(i+8)], byteorder='little'))/10**6
    i += 8
    tail_seconds_ts = float(int.from_bytes(data[i:(i+8)], byteorder='little'))/10**6
    i += 8
    num_frames = int.from_bytes(data[i:(i+2)], byteorder='little')
    i += 2

    gyro_x = data[i:i+num_frames*2]
    gyro_x = [float(
        int.from_bytes(gyro_x[n:n+2], byteorder='little', signed=True))/32768*gyro_dps for n in range(0, num_frames*2, 2)]
    i += num_frames*2

    gyro_y = data[i:i+num_frames*2]
    gyro_y = [float(int.from_bytes(gyro_y[n:n+2], byteorder='little', signed=True))/32768*gyro_dps for n in range(0, num_frames*2, 2)]
    i += num_frames*2

    gyro_z = data[i:i+num_frames*2]
    gyro_z = [float(int.from_bytes(gyro_z[n:n+2], byteorder='little', signed=True))/32768*gyro_dps for n in range(0, num_frames*2, 2)]
    i += num_frames*2

    accel_x = data[i:i+num_frames*2]
    accel_x = [float(int.from_bytes(accel_x[n:n+2], byteorder='little', signed=True))/32768*accel_g_range for n in range(0, num_frames*2, 2)]
    accel_x = np.array(accel_x)
    i += num_frames*2

    accel_y = data[i:i+num_frames*2]
    accel_y = [float(int.from_bytes(accel_y[n:n+2], byteorder='little', signed=True))/32768*accel_g_range for n in range(0, num_frames*2, 2)]
    accel_y = np.array(accel_y)
    i += num_frames*2

    accel_z = data[i:i+num_frames*2]
    accel_z = [float(int.from_bytes(accel_z[n:n+2], byteorder='little', signed=True))/32768*accel_g_range for n in range(0, num_frames*2, 2)]
    accel_z = np.array(accel_z)
    i += num_frames*2

    timestamps = np.linspace(tail_seconds_ts, head_seconds_ts, num_frames+1)[1:]

    return index, np.column_stack([timestamps, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z])

def send_start_command(conn):
    """ Send command to start sampling to ESP32 """
    conn.send(b"start\0")

def send_stop_command(conn):
    """ Send command to stop sampling to ESP32 """
    conn.send(b"stop\0")

def send_cal_command(conn):
    """ Send command to enter calibration moe to ESP32 """
    conn.send(b"cal\0")

def send_ack(conn):
    """ Send acknowledgement to ESP32 """
    conn.send(b"ack\0")
