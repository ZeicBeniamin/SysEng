import numpy as np
import matplotlib.pyplot as plt

sym = "3"

plot_style = [
    f"k{sym}",
    f"g{sym}",
    f"c{sym}",
    f"m{sym}",
    f"y{sym}",
    f"r{sym}",
    f"b{sym}",   
]


plot_style = [
    f"k|",
    f"g_",
    f"c|",
    f"m_",
    f"yx ",
    f"r+",
    f"b.",   
]

def make_plot(x, y, styleno=0, label=""):
    ax.plot(x, y, plot_style[styleno % len(plot_style)], label=label)
    
def spawn_arrays(array, multiplicity=1, noise_mean=0, noise_stddev=0.5, scale=1):
    shape = array.shape[0]
    
    time_scale = 1e-1
    noise = np.random.normal(noise_mean, noise_stddev, size=(shape, multiplicity))
    time_noise = np.random.normal(size=(shape, multiplicity))/3 * time_scale

    noisy = np.expand_dims(array, axis=1) + noise * scale * np.random.uniform(1, 1.5)
    noisy_time = np.expand_dims(np.arange(array.shape[0]), axis=1) + time_noise

    return noisy, noisy_time


arr = np.load("quake2.npy")
noisy, noisy_time = spawn_arrays(arr, multiplicity=10, noise_mean=0, scale=1e-7)

lowerlim = 2750
upperlim = 17500

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for i in range(noisy.shape[1]):
    make_plot(noisy_time[lowerlim:upperlim, i] , noisy[lowerlim:upperlim, i], styleno=i, label=f"W{i}")

# ax.plot(x, y, plot_style[styleno])
plt.legend()
plt.ylabel("meters")
plt.xlabel("microseconds")
plt.title("Displacement - noisy readings")
plt.show()

