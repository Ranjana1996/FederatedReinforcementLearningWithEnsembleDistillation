import matplotlib.pyplot as plt
import numpy as np


def plotgraph(x,avg_x,x_label,y_label,title,file_name,legend=None):
    plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x)
    plt.plot(avg_x)
    if legend:
        plt.legend(legend)
    plt.title(title)
    #plt.show()
    plt.savefig(file_name)


def get_moving_average(x,n):
    y = np.zeros_like(x)
    sm = 0
    count = 0
    for i_n in range(len(x)):
        sm += x[i_n]
        count += 1
        y[i_n] = sm/count
        if count == n:
            sm -= x[i_n - n + 1]
            count = n-1
    return y

