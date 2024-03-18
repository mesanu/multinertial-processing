import socket
import sys
import numpy as np
import yaml
from common import *
import time

NUM_ORIENTATIONS = 6

def main():
    """ Main function for claibration procedure """
    cal_file = open("./calibrations.yaml", "r", encoding="utf-8")
    cal_file.close()

    configs = yaml.safe_load(open("./config.yaml", "r", encoding="utf-8"))

    host = configs["connection"]["host"]
    port = int(configs["connection"]["port"])

    num_frames = int(configs["imu"]["calFrameCount"])
    gyro_dps_range = float(configs["imu"]["gyro"]["dpsRange"])
    accel_g_range = float(configs["imu"]["accel"]["gRange"])
    num_imus = int(configs["imu"]["num_imus"])

    print("connecting")
    esp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    esp_sock.connect((host, port))

    send_cal_command(esp_sock)

    time.sleep(1)

    cal_dicts = []

    orientation_stages = NUM_ORIENTATIONS

    while orientation_stages:
        cal_dicts_orientation = []
        frame_means = [[] for _ in range(num_imus)]
        frame_counts = np.array([num_frames]*num_imus)
        send_start_command(esp_sock)
        while np.any(frame_counts > 0):
            index, imu_frame = get_data_block(esp_sock, gyro_dps_range, accel_g_range)
            if frame_counts[index] > 0:
                frame_means[index].append(np.mean(imu_frame[:,1:], axis=0))
                frame_counts[index] -= 1
            if np.any(frame_counts > 0):
                send_ack(esp_sock)
            else:
                send_stop_command(esp_sock)

        frame_means = np.mean(np.array(frame_means), axis=1)
        for mean in frame_means:
            cal_dict = dict({"gyro":{"x":0, "y":0, "z":0}, "accel":{"x":0, "y":0, "z":0}})
            cal_dict["gyro"]["x"] = float(mean[0])
            cal_dict["gyro"]["y"] = float(mean[1])
            cal_dict["gyro"]["z"] = float(mean[2])
            cal_dict["accel"]["x"] = float(mean[3])
            cal_dict["accel"]["y"] = float(mean[4])
            cal_dict["accel"]["z"] = float(mean[5])
            cal_dicts_orientation.append(cal_dict)

        cal_dicts.append(cal_dicts_orientation)
        print("Finished orientation")
        orientation_stages -= 1

    send_cal_command(esp_sock)

    config_file = open("./calibrations.yaml", "w", encoding="utf-8")
    config_file.write(yaml.dump(cal_dicts))
    config_file.close()

if __name__ == '__main__':
    sys.exit(main())