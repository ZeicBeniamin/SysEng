from linear_kf import LinearKalmanFilterPosVel

lkf = LinearKalmanFilterPosVel()

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

noisy_dis, noisy_vel, noisy_tim = spawn_arrays(dis_arr, vel_arr, multiplicity=10, noise_mean=0, dis_nscale=1e-7, vel_nscale=0)

lowerlim = 2750
upperlim = 17500
# upperlim = 217500

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.legend()
plt.ylabel("meters")
plt.xlabel("microseconds")

# for i in range(noisy_dis.shape[1]):
for i in range(1):
    make_plot(noisy_tim[lowerlim:upperlim, i] , noisy_dis[lowerlim:upperlim, i], styleno=i, label=f"W{i}")

# ax = fig.add_subplot(2, 1, 2)
# plt.ylabel("meters/second")
# plt.xlabel("microseconds")

# for i in range(noisy_vel.shape[1]):
#     make_plot(noisy_tim[lowerlim:upperlim, i] , noisy_vel[lowerlim:upperlim, i], styleno=i, label=f"W{i}")

# ax.plot(x, y, plot_style[styleno])
# plt.title("Displacement - noisy readings")
# plt.show()

displ = noisy_dis[:,1]
veloc = noisy_vel[:,1]
timer = noisy_tim[:,1]

estimated_state = []

for i in range(lowerlim, upperlim):
    measurement = np.array([
        [displ[i]], 
        [veloc[i]]
    ])
    time = timer[i]

    # print(f"Feeding\n {measurement} with time {time}")
    lkf.update_enc_imu(measurement, time=time)
    
    estimated_state.append(lkf.get_state())

est = np.array(estimated_state)

make_plot(noisy_tim[lowerlim:upperlim, 1], est[:,0,0] * 0.07, styleno=3, label=f"Estimation")

plt.legend()
plt.show()

print("Finished")