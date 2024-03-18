import socket
from multiprocessing import Process, Queue, Event
from math import floor
import time
import yaml
import numpy as np
from ursina import *
from common import get_data_block, send_start_command, send_ack
from gradient_descent import get_least_squares_orientation, rotate_vectors_about_axis

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

AVG_INDEX = 0
LEAST_SQUARES_INDEX = 1
AXIS_FLIP = np.array([-1, -1, 1])

class InputWrap(Entity):
    """ Wrapper class to allow for camera control without redefining input() """
    def __init__(self, camera_entity, reset_events, **kwargs):
        super().__init__()
        self.camera_entity = camera_entity
        self.reset_events = reset_events
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
        elif key == 'space':
            for event in self.reset_events:
                event.set()

class AccumulatorEntities:
    """ Container for Ursina entities used for visualizing orientation """
    def __init__(self, center_position, texture):
        self.center_position = center_position * AXIS_FLIP
        self.center_entity = Entity(model='cube', texture=texture, scale=1, position=self.center_position )
        self.basis_x_entity = Entity(position=((-1, 0, 0) + self.center_position ))
        self.basis_y_entity= Entity(position=((0, -1, 0) + self.center_position ))
        self.basis_z_entity= Entity(position=(0, 0, 1) + self.center_position )

        self.center_entity.look_at(self.basis_x_entity, axis='right')
        self.center_entity.look_at(self.basis_y_entity, axis='up')
        self.center_entity.look_at(self.basis_z_entity, axis='forward')

    def update_entities(self, basis_vectors):
        """ Update visualization entities """
        self.basis_x_entity.position = (basis_vectors[0,:] * AXIS_FLIP + self.center_position)
        self.basis_y_entity.position = (basis_vectors[1,:] * AXIS_FLIP + self.center_position)
        self.basis_z_entity.position = (basis_vectors[2,:] * AXIS_FLIP + self.center_position)

        self.center_entity.look_at(self.basis_x_entity, axis='right')
        self.center_entity.look_at(self.basis_y_entity, axis='up')
        self.center_entity.look_at(self.basis_z_entity, axis='forward')


class Accumulator:
    """ Base class for classes the calculate orientation from gyro data """
    def __init__(self, num_imus, render_q, reset_event, render_period, ideal_sample_period, index):
        self.last_raw_data_row = [None for _ in range(num_imus)]
        self.interpolation_leftovers = [[] for _ in range(num_imus)]
        self.next_render_ts =  None

        self.ideal_sample_period = ideal_sample_period

        self.index = index

        self.interpolation_start_ts = None

        self.norm_axes = np.eye(3)

        self.render_q = render_q
        self.reset_event = reset_event
        self.render_period = render_period

        self.gyro_cals = np.zeros((num_imus, 3))

        self.frame_buffers = [[] for _ in range(num_imus)]

        self.num_imus = num_imus

    def _interpolate_frames(self, combined_frames):
        if self.interpolation_start_ts is None:
            self.interpolation_start_ts = np.max(combined_frames[:,0,0])
        least_last_frame_ts = np.min(combined_frames[:,-1, 0])

        num_interpolation_periods = floor((least_last_frame_ts - self.interpolation_start_ts)/self.ideal_sample_period)
        interpolation_end_ts = (self.ideal_sample_period * num_interpolation_periods) + self.interpolation_start_ts

        interpolated_x_points = np.linspace(self.interpolation_start_ts, interpolation_end_ts, num_interpolation_periods + 1)

        interpolated_frames = np.zeros([len(combined_frames), len(interpolated_x_points), 7])

        interpolated_frames[:,:,0] = interpolated_x_points

        for i, frame in enumerate(combined_frames):
            if len(self.interpolation_leftovers[i]) > 0:
                total_frame = np.vstack([self.interpolation_leftovers[i].pop(0), frame])
            else:
                total_frame = frame
            for j in range(1, total_frame.shape[1]):
                interpolated_frames[i,:,j] = np.interp(interpolated_x_points, total_frame[:,0], total_frame[:,j])
            leftovers = total_frame[total_frame[:,0] > interpolation_end_ts]
            if len(leftovers) > 0:
                self.interpolation_leftovers[i].append(leftovers)

        self.interpolation_start_ts = interpolation_end_ts + self.ideal_sample_period
        return interpolated_frames

    def integrate_position_orientation(self, imu_index, frame):
        """ Integrate updated position and/or orientation with IMU data """
        if self.reset_event[self.index].is_set():
            self._reset()
        if self.next_render_ts is None:
            self.next_render_ts = frame[0,0] + self.render_period
        if self.last_raw_data_row[imu_index] is None:
            self.last_raw_data_row[imu_index] = np.copy(frame[-1])
        else:
            frame = np.vstack([self.last_raw_data_row[imu_index], frame])
            self.last_raw_data_row[imu_index] = np.copy(frame[-1])

        cal_frame = frame[:-1]

        cal_frame[:,1] -= self.gyro_cals[imu_index,0]
        cal_frame[:,2] -= self.gyro_cals[imu_index,1]
        cal_frame[:,3] -= self.gyro_cals[imu_index,2]

        self.frame_buffers[imu_index].append(cal_frame)

        for queue in self.frame_buffers:
            if len(queue) == 0:
                return

        next_frame_shape = self.frame_buffers[0][0].shape

        combined_frames = np.zeros([self.num_imus, next_frame_shape[0], next_frame_shape[1]])

        for i,queue in enumerate(self.frame_buffers):
            combined_frames[i] = queue.pop(0)

        interpolated_frames = self._interpolate_frames(combined_frames)

        self._process_interpolated_frame(interpolated_frames)

    def set_calibrations(self, cal_orientations):
        """ Pass calibrations into the accumulator class and set them """
        dim_order = ["x", "y", "z"]
        for i, cal_orientation in enumerate(cal_orientations):
            dim_index = int(i/2)
            dim = dim_order[dim_index]
            for j, imu_cal in enumerate(cal_orientation):
                if j >= self.num_imus:
                    break
                self.gyro_cals[j] += np.array([float(imu_cal["gyro"][dim]), float(imu_cal["gyro"][dim]), float(imu_cal["gyro"][dim])])

        self.gyro_cals /= len(cal_orientations)

class AverageAccumulator(Accumulator):
    """ Accumulator that calculates orientation by averaging data from IMUs """

    def _process_interpolated_frame(self, interpolated_frames):
        avg_frame = np.mean(interpolated_frames[:,:,1:], axis=0)
        gyro_data = avg_frame[:,:3]

        rates_of_rotation = np.linalg.norm(gyro_data, axis=1)
        axes_of_rotation = gyro_data/rates_of_rotation[:, None]

        rotation_angles = rates_of_rotation*self.ideal_sample_period
        for i in range(len(rates_of_rotation)):
            self.norm_axes = rotate_vectors_about_axis(self.norm_axes, np.matmul(self.norm_axes.T, axes_of_rotation[i]), rotation_angles[i])
            if interpolated_frames[0,i,0] > self.next_render_ts:
                self.render_q.put([self.index, np.copy(self.norm_axes)])
                self.next_render_ts += self.render_period

    def _reset(self):
        self.norm_axes = np.eye(3)
        self.reset_event[self.index].clear()

class LeastSquaresAccumulator(Accumulator):
    """ Accumulator that calculates orientation by finding least squares fit between IMUs """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imu_norm_axes = np.array([np.eye(3), np.eye(3), np.eye(3)])

    def _process_interpolated_frame(self, interpolated_frames):
        gyro_data = interpolated_frames[:,:,1:4]

        rates_of_rotation = np.linalg.norm(gyro_data, axis=2)
        axes_of_rotation = gyro_data/rates_of_rotation[:,:,None]

        rotation_angles = rates_of_rotation*self.ideal_sample_period
        for i in range(rates_of_rotation.shape[1]):
            for j in range(self.num_imus):
                self.imu_norm_axes[j] = rotate_vectors_about_axis(self.imu_norm_axes[j], np.matmul(self.imu_norm_axes[j].T, axes_of_rotation[j,i]), rotation_angles[j,i])
            if interpolated_frames[0,i,0] > self.next_render_ts:
                self.norm_axes, updated = get_least_squares_orientation(self.imu_norm_axes[1], self.imu_norm_axes, 80, 0.1)
                if updated:
                    self.imu_norm_axes = np.array([self.norm_axes] * self.num_imus)
                self.render_q.put([self.index, np.copy(self.norm_axes)])
                self.next_render_ts += self.render_period

    def _reset(self):
        self.norm_axes = np.eye(3)
        self.imu_norm_axes = np.array([np.eye(3), np.eye(3), np.eye(3)])
        self.reset_event[self.index].clear()

def visualize(render_q, reset_events,):
    """ Subprocess function for displaying and updating entities """
    ursina_app = Ursina(borderless = False)
    input_wrap = InputWrap(camera, reset_events)
    accumulator_models = [
        AccumulatorEntities((-2, 0, 0), 'brick'),
        AccumulatorEntities((2, 0, 0), 'brick')
    ]

    error_display_strings = [
        "Average error from start: {:.4f}",
        "Least squares error from start: {:.4f}"
    ]

    error_displays = [
        Text(text=error_display_strings[0].format(0), position=(-0.8, 0.3)),
        Text(text=error_display_strings[1].format(0), position=(-0.8, 0.2))
    ]

    origin = Entity(position=(0,0,0))

    camera.position = (0, 25, 0)
    camera.look_at(origin)
    camera.rotate((0, 0, 180))
    while True:
        data = render_q.get()
        index = data[0]
        basis_vectors = data[1]
        accumulator_models[index].update_entities(basis_vectors)
        error_from_start = np.mean(
            np.arccos(np.clip(np.sum(basis_vectors * np.eye(3), axis=1),-1.0,1.0))
        )
        error_displays[index].text = error_display_strings[index].format(error_from_start)
        ursina_app.step()

def main():
    """ Main function for visualization """
    configs = yaml.safe_load(open("./config.yaml", "r", encoding="utf-8"))

    host = configs["connection"]["host"]
    port = int(configs["connection"]["port"])
    accel_g_range = float(configs["imu"]["accel"]["gRange"])
    gyro_dps_range = float(configs["imu"]["gyro"]["dpsRange"])

    num_imus = int(configs["imu"]["numImus"])

    ideal_sample_period = float(configs["imu"]["idealSamplePeriod"])
    vis_update_period = float(configs["visualizer"]["renderPeriod"])

    calibrations = yaml.safe_load(open("./calibrations.yaml", encoding="utf-8"))

    render_q = Queue()
    reset_events = [Event(), Event()]

    avg_accumulator = AverageAccumulator(
        num_imus,
        render_q,
        reset_events,
        vis_update_period,
        ideal_sample_period,
        AVG_INDEX
        )
    avg_accumulator.set_calibrations(calibrations)

    least_squares_accumulator = LeastSquaresAccumulator(
        num_imus,
        render_q,
        reset_events,
        vis_update_period,
        ideal_sample_period,
        LEAST_SQUARES_INDEX
    )
    least_squares_accumulator.set_calibrations(calibrations)

    render_p = Process(target=visualize, args=(render_q, reset_events,))

    render_p.start()

    time.sleep(2)

    print("connecting")
    esp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    esp_sock.connect((host, port))
    esp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
    send_start_command(esp_sock)
    while True:
        imu_index, imu_frame = get_data_block(esp_sock, gyro_dps_range, accel_g_range)
        avg_accumulator.integrate_position_orientation(imu_index, imu_frame)
        least_squares_accumulator.integrate_position_orientation(imu_index, imu_frame)
        send_ack(esp_sock)

if __name__ == '__main__':
    sys.exit(main())
