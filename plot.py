import numpy as np
import matplotlib.pyplot as plt

def make_plot(x, y, title='', ylab=''):
    '''
    input: trace
    
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, "k-")
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()

arr = np.load("quake2.npy")

make_plot(np.arange(arr.shape[0]), arr, title="Ground displacement (m)")

# print(arr.shape)
