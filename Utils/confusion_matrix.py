from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(label, pred, labels, title='Confusion Matrix', cmap=plt.cm.binary):
    fsz=20
    cm = confusion_matrix(label, pred)
    np.set_printoptions(precision=2)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fsz)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fsz)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90, fontsize=fsz)
    plt.yticks(xlocations, labels, fontsize=fsz)
    plt.show