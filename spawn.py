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
    f"b-",   
]

def make_plot(x, y, styleno=0, label=""):
    ax.plot(x, y, plot_style[styleno % len(plot_style)], label=label)
    
def spawn_arrays(dis_array, vel_array, multiplicity=1, noise_mean=0, noise_stddev=0.5, dis_nscale=1, vel_nscale=1):
    shape = dis_array.shape[0]
    
    time_scale = 1e-1
    noise = np.random.normal(noise_mean, noise_stddev, size=(shape, multiplicity))
    time_noise = np.random.normal(size=(shape, multiplicity))/3 * time_scale

    noisy_dis = np.expand_dims(dis_array, axis=1) + noise * dis_nscale * np.random.uniform(1, 1.5)
    noisy_vel = np.expand_dims(vel_array, axis=1) + noise * vel_nscale * np.random.uniform(1, 1.5)
    noisy_time = np.expand_dims(np.arange(dis_array.shape[0]), axis=1) + time_noise

    return noisy_dis, noisy_vel, noisy_time


dis_arr = np.load("quake_dis2.npy")
vel_arr = np.load("quake_vel2.npy")

multiplicity=1
noisy_dis, noisy_vel, noisy_time = spawn_arrays(dis_arr, vel_arr, multiplicity=multiplicity, noise_mean=0, dis_nscale=0, vel_nscale=0)

lowerlim = 2750
upperlim = 17500

lowerlim = 0
upperlim = 17500000
# upperlim = 217500

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
# ax.legend()

for i in range(noisy_dis.shape[1]):
    make_plot(noisy_time[lowerlim:upperlim, i] , noisy_dis[lowerlim:upperlim, i], styleno=6, label=f"Ground displacement")


ax.legend()
plt.ylabel("meters")
plt.xlabel("microseconds")


abs_dis_post = np.expand_dims(np.append(np.abs(noisy_dis), 0), axis=1)
abs_dis_pre = np.insert(np.abs(noisy_dis), 0, 0, axis=0)

halves = (abs_dis_post * abs_dis_pre / 2)

time_post = np.expand_dims(np.append(noisy_time, 0), axis=1)
time_pre = np.insert(noisy_time, 0, 0, axis=0)

time_delta = time_post - time_pre

part_integral = halves * time_delta


cumsum = np.cumsum(part_integral)[1:]

cumsum2 = np.array([], dtype=np.float64)
stride = 10000
for i in range(lowerlim, upperlim, stride):
    cumsum2 = np.append(cumsum2, np.sum(part_integral[i:i + stride]))

# for i in range(lowerlim, upperlim):
#     accum_dis[i:] += np.trapezoid(abs_dis[lowerlim:upperlim], noisy_time[lowerlim:upperlim], axis=0)

ax = fig.add_subplot(2, 1, 2)
# plt.ylabel("meters/second")
# plt.xlabel("microseconds")

# for i in range(noisy_vel.shape[1]):
# make_plot(noisy_time[lowerlim:upperlim, i] , cumsum[lowerlim:upperlim] , styleno=i, label=f"W{i}")
ax.plot(cumsum2)

# ax.plot(x, y, plot_style[styleno])
# plt.title("Displacement - noisy readings")
plt.show()

