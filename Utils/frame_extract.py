import cv2
import os
import threading
import numpy as np
 
def video_to_frames(video_path, outPutDirName, write = True):
    times = 0
    end_frame = 10
    frame_frequency = 100

    if write and not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)
         
    camera = cv2.VideoCapture(video_path)
    similarity = []

    while True:
        times = times + 1
        res, image = camera.read()
        if times == 1:
            start_image = np.array(image)
        if not res:
            break
        if not write:
            image = np.array(image)
            new_start_img = start_image - start_image.sum()/(image.shape[0]*image.shape[1]*image.shape[2])
            new_img = image - image.sum()/(image.shape[0]*image.shape[1]*image.shape[2])
            correlation = (new_start_img * new_img).sum() / ((new_start_img*new_start_img).sum()**0.5 * (new_img * new_img).sum()**0.5)

            similarity.append(correlation)
        if write and (times <= end_frame or ((times % frame_frequency) == 0 and times < 300)):
            cv2.imwrite(outPutDirName + '/' + str(times)+'.jpg', image)
             
    camera.release()
    return similarity


if __name__ == "__main__":
    input_dir = r'./10uL/'     
    save_dir = r'./10uL/Frames'    
    count = 0   # 视频数
    for video_name in os.listdir(input_dir):
        if video_name[-4:] == ".mp4":
            video_path = os.path.join(input_dir, video_name)
            outPutDirName = os.path.join(save_dir, video_name[:-4])
            threading.Thread(target=video_to_frames, args=(video_path, outPutDirName)).start()
            count = count + 1