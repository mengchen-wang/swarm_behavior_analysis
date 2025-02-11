import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from Utils.frame_extract import video_to_frames
import warnings
import numpy as np
import json

def plot_similarity(_, SimilarityDirName):
    warnings.filterwarnings('ignore')
    if not os.path.exists(SimilarityDirName):
        os.makedirs(SimilarityDirName)
    input_dir = r'../10uL/' 
    video_name = "%d.mp4"%_
    video_path = os.path.join(input_dir, video_name)
    similarity = video_to_frames(video_path, False, False)
    plt.figure()
    plt.plot(similarity)
    plt.savefig(SimilarityDirName + "similarity of parameter %d"%_)
    return similarity

if __name__ == "__main__":
    SimilarityDirName = "../fig/similarity/"
    X_1 = np.load("../Data/X_1.npy")
    y = np.load("../Data/y.npy")
    similarity_dict = {}
    tf = open("../Data/myDictionary.json", "w")
    for index, _ in enumerate(X_1):
        similarity = plot_similarity(_,SimilarityDirName+"%d/"%y[index])
        similarity_dict['%d'%_] = similarity
    json.dump(similarity_dict,tf)
    tf.close()
        
