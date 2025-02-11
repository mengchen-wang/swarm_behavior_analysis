import os
import matplotlib.pyplot as plt
figure_path = 'fig'
pattern_dict= ["Liquid", "Network", "Thread", "Flocculence", "Vortex", "Rod"]

def plot_2d_visualization(data, y_train, method: str, n_neighbour=-1, dist=-1, datatype = "augmented", s=1):
    plt.figure(figsize=(4, 4))
    # plt.xlabel('%s 1' % method, fontsize=10)
    # plt.ylabel('%s 2' % method, fontsize=10)
    # axes = plt.gca()
    # axes.xaxis.label.set_size(16)
    # axes.yaxis.label.set_size(16)
    fsz = 20
    plt.scatter(data[:, 0], data[:, 1], s, c=y_train)
    plt.tick_params(axis='both', which='major', labelsize=fsz)
    plt.xlabel('PCA 1', fontsize=fsz)
    plt.ylabel('PCA 2', fontsize=fsz)
    if n_neighbour == -1 and datatype == "augmented":
        plt.savefig(os.path.join(figure_path, "%s.png" % method),bbox_inches="tight", dpi=300)
    elif n_neighbour == -1: 
        plt.savefig(os.path.join(figure_path, "%s_original.png" % method),bbox_inches="tight", dpi=300)
    elif datatype == "augmented": 
        plt.savefig(os.path.join(figure_path, "%s_n_neighbour=%d_dist=%.02f.png" % (method, n_neighbour,float(dist)/10)),bbox_inches="tight", dpi=300)
    else:
        plt.savefig(os.path.join(figure_path, "%s_original_n_neighbour=%d_dist=%.02f.png" % (method, n_neighbour,float(dist)/10)),bbox_inches="tight", dpi=300)
