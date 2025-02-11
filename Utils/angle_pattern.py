from numba import njit
import numpy as np
import os
import cv2
from io_utils import index_to_parameter
import matplotlib.pyplot as plt

@njit
def clip(x, minv, maxv):
    if x > maxv:
        return maxv
    if x < minv:
        return minv
    return x

@njit
def angle_pattern(fx, fy, fz, bx, by, bz, phi_x, phi_y, phi_z, t, frame_num):
    tps = np.linspace(0, t, int(frame_num))
    print(tps[1])
    angles = np.zeros(tps.shape[0])
    for i, _ in enumerate(tps):
        B1 = np.array([bx*np.sin(phi_x), by*np.sin(phi_y*np.pi/6), bz*np.sin(phi_z*np.pi/2)])
        B2 = np.array([bx*np.sin(2*np.pi*fx*_+phi_x), by*np.sin(2*np.pi*fy*_+phi_y*np.pi/6), bz*np.sin(2*np.pi*fz*_+phi_z*np.pi/2)])
        if np.sqrt((B1*B1).sum())*np.sqrt((B2*B2).sum()) == 0:
            theta = 0
        else:
            theta = np.arccos(clip((B1*B2).sum()/(np.sqrt((B1*B1).sum())*np.sqrt((B2*B2).sum())),-1,1))
        angles[i]=theta 
    return angles

if __name__ == "__main__":
    input_dir = r'./10uL/' 
    angle_dir = "./fig/angles/"
    if not os.path.exists(angle_dir):
        os.makedirs(angle_dir)
    for video_name in os.listdir(input_dir):
        if video_name[-4:] == ".mp4":
            video_path = os.path.join(input_dir, video_name)
            camera = cv2.VideoCapture(video_path)
            fps = camera.get(cv2.CAP_PROP_FPS)
            frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
            t = frames/fps
            X = index_to_parameter(int(video_name[:-4]))
            angles = angle_pattern(X[0], X[1], X[2],X[3], X[4], X[5], 0, X[6], X[7], t, frames)
            camera.release()
            plt.figure()
            plt.plot(angles)
            plt.savefig(angle_dir+"angle for parameter %d"%int(video_name[:-4]))