import math
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

figure_path = 'fig'
d = 14
pattern_dict= ["Liquid", "Network", "Ribbon", "Flocculence", "Vortex", "Rod"]
neighbourSize = 2.2
_lambda = 1

def compute_dist(X1, X2, _type = 'l2'):
    if _type == 'l2':
        dist = 0
        for i in range(len(X1)):
            dist += (X1[i]-X2[i])*(X1[i]-X2[i])
        return math.sqrt(dist)
    else:
        return max(X1-X2)

def cluster_distance(class1, class2):
    max = 1e5
    min_x1 = min_x2 =0
    for i in range (X_train.shape[0]):
        if y_train[i] != class1:
            continue
        for j in range (X_train.shape[0]):
            if y_train[j] != class2:
                continue
            x_1 = X_train[i]
            x_2 = X_train[j]
            if compute_dist(x_1, x_2) < max:
                max = compute_dist(x_1, x_2)
                min_x1 = x_1
                min_x2 = x_2
    return min_x1, min_x2, max 
    
def stability(i, X_train, y_train):
    max = 1e5
    x_1 = X_train[i]
    for j in range (X_train.shape[0]):
        x_2 = X_train[j]
        if y_train[i] != y_train[j] and compute_dist(x_1, x_2) < max:
            max = compute_dist(x_1, x_2)
            min_x2 = x_2
    return min_x2, max 

def stability_list(X_train, y_train):
    stabilityList = []
    contrastList = []
    for i in range(X_train.shape[0]):
        _, __ = stability(i, X_train, y_train)
        contrastList.append(_)
        stabilityList.append(__)
    return contrastList, stabilityList

def stability2(X_train, y_train, dist):
    stabilityList = []
    for i in range(X_train.shape[0]):
        count, contrast = 0.0, 0.0
        for j in range(X_train.shape[0]):
            if compute_dist(X_train[i], X_train[j]) <= dist:
                count += 1
                if y_train[i] == y_train[j]:
                    contrast += 1
        stabilityList.append(contrast/count)
    return stabilityList

def plot_stability(X_train, stabilityList):
    plt.figure(figsize=(9.6, 7.2))
    plt.plot(range(X_train.shape[0]), stabilityList)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Stability score', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, "stability.png"))

def draw_stability_distribution(stabilityList, d:int):
    plt.figure(figsize=(9.6, 7.2))
    train_data = np.digitize(stabilityList, np.linspace(min(stabilityList), max(stabilityList), d))
    plt.hist(train_data, np.arange(d) - 0.5, density=True)
    plt.xlabel('Digitized stability',  fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, "stability_sub_0.png"))
    return train_data

def draw_pattern_stability(X, i:int):
    plt.figure(figsize=(4, 4)) 
    plt.title('%s' % pattern_dict[i], fontsize=28)
    plt.ylim(0, 0.37)
    plt.hist(X[i], bins=np.arange(d) - 0.5, density=True, color='#617cb8')
    plt.tick_params(axis='both', which='major', labelsize=28)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, "stability_%s.png" % pattern_dict[i]))

def get_pattern_stability_distribution(index:int, train_data, y_train, stabilityList):
    X,Y = [],[]
    for i in range(len(y_train)):
        if y_train[i] == index:
            X.append(train_data[i])
            Y.append(stabilityList[i])
    return X,Y

def add_text(bars, err, width, fontsize=8):
        for index, bar in enumerate(bars):
            x = bar.get_x() + width/2
            h = bar.get_height()
            plt.text(x, h+err[index]+0.1, round(h+err[index],1), color='k', fontsize=fontsize, ha='center')
            plt.text(x, h-err[index]-0.15, round(h-err[index],1), color='w', fontsize=fontsize, ha='center', va='bottom')

def plot_error_graph(list0, err0, list1, err1, list2, err2, neighbourSize):
    print(list0, err0, list1, err1, list2, err2)
    width=0.3
    fsz = 24
    index = np.arange(len(list0))	
    plt.figure(figsize=(15, 9))
    plt.ylim(0, 7.5, )
    plt.tick_params(axis='both', which='major', labelsize=fsz)
    bar0 = plt.bar(index-width, np.array(list0), width, color='#3f3b72', yerr=err0)
    bar1 = plt.bar(index, np.array(list1), width,  color='#617cb8', yerr=err1)
    bar2 = plt.bar(index+width, np.array(list2), width, color='#df9b92', yerr=err2)
    plt.ylabel('Stability score', fontsize=fsz)
    plt.xticks(index, pattern_dict, fontsize=fsz)
    plt.legend([bar0, bar1, bar2], ['Integrated stability', 'Transformation stability', '%.1f neighbourhood stability' % neighbourSize], fontsize=fsz)
    add_text(bar0, err0, width, fontsize=fsz)
    add_text(bar1, err1, width, fontsize=fsz)
    add_text(bar2, err2, width, fontsize=fsz)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, "stability_std_%.1f.png" % neighbourSize))

def stability_experiment(X_train_pre, y_train_pre, _lambda, neighbourSize, showresult=False):
    contrastList, stabilityList = stability_list(X_train_pre, y_train_pre)
    stabilityList2 = stability2(X_train_pre, y_train_pre, neighbourSize)
    stabilityList = np.array(stabilityList)
    stabilityList2 = np.array(stabilityList2)
    mean1 = stabilityList.mean()
    mean2 = stabilityList2.mean()
    integratedStabilityList = _lambda * ((stabilityList-mean1)/stabilityList.std()) + ((stabilityList2-mean2)/stabilityList2.std())
    integratedStabilityList -= min(integratedStabilityList)
    plot_stability(X_train_pre, integratedStabilityList)
    train_data = draw_stability_distribution(integratedStabilityList, d)
    hist_stability_pattern, stability_pattern, stability1_pattern, stability2_pattern  = [],[],[],[]
    for i in range (6):
        x,y = get_pattern_stability_distribution(i, train_data, y_train_pre, integratedStabilityList)
        _,y1 = get_pattern_stability_distribution(i, train_data, y_train_pre, stabilityList)
        _,y2 = get_pattern_stability_distribution(i, train_data, y_train_pre, stabilityList2)
        hist_stability_pattern.append(x)
        stability_pattern.append(y)
        stability1_pattern.append(y1)
        stability2_pattern.append(y2)
    for i in range(6):
        draw_pattern_stability(hist_stability_pattern, i)
    list0, err0, list1, err1, list2, err2 = [],[],[],[],[],[]
    for i in range (6):
        list0.append(np.array(stability_pattern[i]).mean())
        err0.append(np.array(stability_pattern[i]).std())
        list1.append(np.array(stability1_pattern[i]).mean())
        err1.append(np.array(stability1_pattern[i]).std())
        list2.append(np.array(stability2_pattern[i]).mean())
        err2.append(np.array(stability2_pattern[i]).std())
    plot_error_graph(list0, err0, list1, err1, list2, err2, neighbourSize)

if __name__ == "__main__":
    X_train_pre = np.load("./Data/X_train_pre.npy")
    X_train_pre = StandardScaler().fit_transform(X_train_pre)
    y_train_pre = np.load("./Data/y_train_pre.npy")
    stability_experiment(X_train_pre, y_train_pre, _lambda, neighbourSize)