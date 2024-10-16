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
    f"m1",
    f"yx ",
    f"r1",
    f"b.",
]

def make_plot(x, y, styleno=0, label=""):
    ax.plot(x, y, plot_style[styleno % len(plot_style)], label=label)

def make_plot_special(x, y, plot_style, label=""):
    ax.plot(x, y, plot_style, label=label)
    
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

working_sensors = 10

noisy_dis, noisy_vel, noisy_tim = spawn_arrays(dis_arr, vel_arr, multiplicity=working_sensors, noise_mean=0, dis_nscale=1e-7, vel_nscale=0)

lowerlim = 2750
upperlim = 17500
lowerlim = 6000
upperlim = 8000
# upperlim = 217500

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)

for i in range(noisy_dis.shape[1]):
# for i in range(1):
    make_plot(noisy_tim[lowerlim:upperlim, i] , noisy_dis[lowerlim:upperlim, i], styleno=i, label=f"W{i}")
ax.legend()
plt.title(f"Noisy measurements - {working_sensors} sensors")
plt.ylabel("meters")
plt.xlabel("microseconds")

# ax = fig.add_subplot(2, 1, 2)
# plt.ylabel("meters/second")
# plt.xlabel("microseconds")

# for i in range(noisy_vel.shape[1]):
#     make_plot(noisy_tim[lowerlim:upperlim, i] , noisy_vel[lowerlim:upperlim, i], styleno=i, label=f"W{i}")

# ax.plot(x, y, plot_style[styleno])
# plt.title("Displacement - noisy readings")
# plt.show()

# displ = noisy_dis[:,1]
# veloc = noisy_vel[:,1]
# timer = noisy_tim[:,1]

# noisy_dis - N, 10           10, N           (1, N
# niosy_tim - N, 10  =>>>>    10, N  =>>>>>    1, N            === > (3, N), (3, N), ... , (3, N) => sort (3,:) 
# noisy_vel - N, 10           10, N            1, N) // x10 x 3

reshaped = np.array([[],[],[]])
# reshaped = np.expand_dims(reshaped, axis=0)
# reshaped = np.expand_dims(reshaped, axis=0)

for i in range(noisy_tim.shape[1]):
    newvec = np.concatenate(
        (np.expand_dims(noisy_dis[:,i], axis=0),
        np.expand_dims(noisy_vel[:,i], axis=0),
        np.expand_dims(noisy_tim[:,i], axis=0)),

    )
    reshaped = np.concatenate((reshaped, newvec), axis=1)

sort_idxes = np.argsort(reshaped)

reshaped = reshaped[:,sort_idxes[2, :]]

ax2 = 1e-7
lkf.set_ax2(ax2)
R_enc = np.array(
        [[1e-6, 0],
        [0, 1e-2]], dtype=np.float32
    )
lkf.set_R_enc(R_enc)

estimated_state = np.zeros((3, reshaped.shape[1]))
for i in range(upperlim * 10):
    measurement = np.array([
        [reshaped[0, i]], 
        [reshaped[1, i]]
    ])
    time = reshaped[2, i]

    # print(f"Feeding\n {measurement} with time {time}")
    lkf.update_enc_imu(measurement, time=time)
    
    estimated_state[0, i] = lkf.out_x[0, 0]
    estimated_state[1, i] = lkf.out_x[1, 0]
    estimated_state[2, i] = time

ax = fig.add_subplot(2, 1, 2)
ax.legend()
plt.title("Estimated signal")
make_plot_special(estimated_state[2, lowerlim*working_sensors:upperlim*working_sensors], estimated_state[0, lowerlim*working_sensors:upperlim*working_sensors], plot_style="g-", label=f"Estimation")
make_plot_special(noisy_tim[lowerlim:upperlim, 1], dis_arr[lowerlim:upperlim], plot_style="r-", label=f"Original")
plt.ylabel("meters")
plt.xlabel("microseconds")

plt.legend()
plt.show()

print("Finished")