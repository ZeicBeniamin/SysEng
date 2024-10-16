import time

import numpy as np


def _multi_dot_three(A, B, C, out=None):
    """
    Dot product of 3 vectors/matrices in a way that is the most efficient.

    For three arguments `_multi_dot_three` is approximately 15 times faster
    than `_multi_dot_matrix_chain_order`

    :param A: First vector/matrix
    :param B: Second vector/matrix
    :param C: Third vector/matrix
    :param out: Where to store output of multiplication, defaults to None
    :return: Dot product of the three vectors/matrices
    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return np.dot(np.dot(A, B), C, out=out)
    else:
        return np.dot(A, np.dot(B, C), out=out)


class LinearKalmanFilterPosVelAcc:
    """
    Models a Kalman filter that tracks 6 parameters (position, velocity,
    acceleration) on both x and y axes.

    .. TODO: The current implementation uses a 6 state system (px, vx, ax, py,
        vy, ay). Thus, we have a 6x6 uncertainty matrix (and other large
        matrices) that contribute to long matrix multiplication times.
        A better implementation would split the system in two 3-state
        subsystems, one containing position, velocity and acceleration
        values for the x axis, and one for the y axis.

    The different 'update' methods are called when sensor data is received.
    Each 'update' method internally calls the 'predict()' method first, which
    updates the state vector `x`. Then, the update step is run, which again
    updates state vector `x` (but this time with values from the measurements).
    After these two steps (predict & update), the x and y coordinates from the
    state vector are copied to the outptut vector, `x`.

                ┌───────────────────────┐
                │                       │
                │                       │
                │                       │
    measurement │      Linear           │       x
    ───────────►│--┐   Kalman       ┌---├────────►
                │  |   Filter       |   │
                │  |                |   │
                │  └---> update()---┘   │
                │     [&predict()]      │
                │                       │
                └───────────────────────┘

    NOTE: Each input/output has an "in"/"out" prefix. The comments will use
    `out_variable` and `variable` interchangeably to refer to the same output
    variable.

    Input
    -----

    Output
    ------
    `out_x`: State vector estimated by the Kalman filter
        The vector contains 6 states: [px, vx, ax, py, vy, ay]
    `out_P`: Covariance matrix of size NxN, where N is the size of `out_x`
        The diagonal of this matrix gives a rough estimate of the probability
        of the corresponding states from `out_x`
    """

    def __init__(self, x_init=None, P_init=None):
        """
        Initialize the state and uncertainty matrices of the Kalman filter.

        :param x_init: Initial state vector (1 x 6 matrix)
        :param P_init: Initial uncertainty (6x6 matrix)
        """
        if x_init is not None:
            self.out_x = np.array(x_init, dtype=np.float32)
            assert self.out_x.size == 4, "Initial state vector must have 6 elements"
        else:
            self.out_x = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float32).T
        if P_init is not None:
            self.P = np.array(P_init, dtype=np.float32)
        else:
            self.P = np.array(
                [[0.1, 0, 0, 0, 0, 0],
                 [0, 100, 0, 0, 0, 0],
                 [0, 0, 100, 0, 0, 0],
                 [0, 0, 0, 0.1, 0, 0],
                 [0, 0, 0, 0, 100, 0],
                 [0, 0, 0, 0, 0, 100]], dtype=np.float32
            )

        # Squared acceleration noises in each direction
        self.ax2 = 0
        self.ay2 = 0
        # Identity matrix
        self.I_6x6 = np.eye(len(self.out_x))
        # Measurement uncertainty - encoder & IMU
        self.R_enc = np.array(
            [[1, 0],
             [0, 1]], dtype=np.float32
        )

        # Measurement uncertainty - GPS
        self.R_gps = np.array(
            [[1, 0],
             [0, 1]], dtype=np.float32
        )

        # Measurement uncertainty - camera
        self.R_cam = np.array(
            [[1, 0],
             [0, 1]], dtype=np.float32
        )

        # Measurement function - encoder
        self.H_enc = np.array(
            [[0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0]], dtype=np.float32
        )

        # Measurement function - GPS
        self.H_gps = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0]], dtype=np.float32
        )

        # Measurement function - camera
        self.H_cam = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0]], dtype=np.float32
        )

        # Preallocate matrices for certain multiplication operations.
        self.S_pre = np.zeros((2, 2), dtype=np.float32)
        self.K_pre = np.zeros((6, 2), dtype=np.float32)
        self.ikh = np.zeros((6, 6), dtype=np.float32)
        self.preal_a = np.zeros((6, 6), dtype=np.float32)
        self.preal_b = np.zeros((6, 6), dtype=np.float32)

        # Flags
        self._is_initialized_enc = False
        self._is_initialized_gps_or_camera = False
        self._last_measurement_time = None

    def set_R_enc(self, R_enc):
        self.R_enc = R_enc

    def set_R_gps(self, R_gps):
        self.R_gps = R_gps

    def set_R_cam(self, R_cam):
        self.R_cam = R_cam

    def set_ax2(self, ax2):
        self.ax2 = ax2

    def set_ay2(self, ay2):
        self.ay2 = ay2
    
    def set_acc_noise(self, ax2, ay2):
        self.ax2 = ax2
        self.ay2 = ay2

    def update_enc_imu(self, measurement):
        """
        Update the filter with values coming from the encoder and IMU.

        This is only a wrapper method. Internally, it calls another method,
        ``_update_enc_imu``, that executes the update.

        :param measurement: Vector containing two values, vx and vy, giving
        the speed on each of the vx and vy axes.
        :return:
        """

        # If the filter was not initialized, initialize it with the first
        # velocity measurement.
        if not self._is_initialized_enc:
            self._last_measurement_time = time.time()
            self.out_x[1] = measurement[0]
            self.out_x[4] = measurement[1]
            self._is_initialized_enc = True
            return
        # Compute the time interval between last and current measurement
        dt = time.time() - self._last_measurement_time
        # Store the time of the current measurement
        self._last_measurement_time = time.time()
        # Predict, then update
        self.get_predicted_lkf(dt)
        self._update_lkf_enc_imu(measurement)

    def update_gps(self, measurement):
        """
        Update the filter with values coming from the GPS.

        This is only a wrapper method. Internally, it calls another method,
        ``_update_gps``, that executes the update.

        :param measurement: Vector containing two values, px and py, giving
        the position on both x and y axes.
        :return:
        """
        if not self._is_initialized_gps_or_camera:
            # GPS and camera will have the same init time, because they
            # initialize  the same parameters (vx, vy) AND because we only do
            # this initialization from one source (either gps, or camera).
            self._last_measurement_time = time.time()
            self.out_x[0] = measurement[0]
            self.out_x[3] = measurement[1]
            self._is_initialized_gps_or_camera = True
            return
        # Compute the time interval between last and current measurement
        dt = time.time() - self._last_measurement_time
        # Store the time of the current measurement
        self._last_measurement_time = time.time()
        # Predict, then update
        self.get_predicted_lkf(dt)
        self._update_lkf_gps(measurement)

    def update_camera(self, measurement):
        """
        Update the filter with measurements coming from the camera.

        This is only a wrapper method. Internally, it calls another method,
        ``_update_camera``, that executes the update.

        :param measurement: Vector containing two values, px and py, giving
        the position on both x and y axes.
        :return:
        """

        if not self._is_initialized_gps_or_camera:
            # GPS and camera will have the same init time, because they
            # initialize  the same parameters (vx, vy) AND because we only do
            # this initialization from one source (either gps, or camera).
            self._last_measurement_time = time.time()
            self.out_x[0] = measurement[0]
            self.out_x[3] = measurement[1]
            self._is_initialized_gps_or_camera = True
            return
        # Compute the time interval between last and current measurement
        dt = time.time() - self._last_measurement_time
        # Store the time of the current measurement
        self._last_measurement_time = time.time()
        # Predict, then update
        self.get_predicted_lkf(dt)
        self._update_lkf_camera(measurement)

    def get_predicted_lkf(self, dt):
        """
        Predict the state of the system after a given period of time.

        Updates the x (state) vector, together with the uncertainity matrix
        (P) based on the time `dt` passed since the last measurement.

        :param dt: Time interval passed from the last measurement.
        :return: None
        """
        # dt == 0 means two measurements came with the same timestamp, one of them 
        # has already been used for updating the LKF, but now the second one
        # also tries to update the LKF.
        # It will make the predict() & update() calls, but in this case 
        # predict() isn't needed, since a previous measurement already 
        # trigerred a predict() step for the same timestamp.
        if dt == 0:
            return

        t = dt
        # Frequent terms in matrices
        t2 = t ** 2
        t3 = t ** 3
        t4 = t ** 4

        # Process noise matrix
        Q = np.array(
            [[t4 / 4 * self.ax2,   t3 / 2 * self.ax2, t2 / 2 * self.ax2, 0,                 0,                  0                   ],
             [t3 / 2 * self.ax2,   t2 * self.ax2,     t * self.ax2,      0,                 0,                  0                   ],
             [t2 / 2 * self.ax2,   t * self.ax2,      1 * self.ax2,      0,                 0,                  0                   ],
             [0,        0,      0,                                      t4 / 4 * self.ay2,  t3 / 2 * self.ay2,  t2 / 2 * self.ay2   ],
             [0,        0,      0,                                      t3 / 2 * self.ay2,  t2 * self.ay2,      t * self.ay2        ],
             [0,        0,      0,                                      t2 / 2 * self.ay2,  t * self.ay2,       1 * self.ay2        ]],
            dtype=np.float32) 
        # Transition matrix
        F = np.array(
            [[1,    t,  t2 / 2, 0,  0,  0],
             [0,    1,  t,      0,  0,  0],
             [0,    0,  1,      0,  0,  0],
             [0,    0,  0,      1,  t,  t2 / 2],
             [0,    0,  0,      0,  1,  t],
             [0,    0,  0,      0,  0,  1]],
            dtype=np.float32
        )

        # Calculate the new state of the system, based on the previous state.
        # x' = F * x

        # # Set acceleration to zero
        # self.out_x[2][0] = 0
        # self.out_x[5][0] = 0

        self.out_x = np.matmul(F, self.out_x, dtype=np.float32)
        # Calculate the covariance matrix (uncertainty estimate)
        # P' = F * P * F.T + Q
        
        # Set acceleration to zero
        # self.out_x[2][0] = 0
        # self.out_x[5][0] = 0
        

        self.P = _multi_dot_three(F, self.P, F.T)
        self.P = self.P + Q

    def _update_lkf_enc_imu(self, z):
        """
        Update the filter with measurements from the encoder and the imu.

        :param z: Same as the `measurement` parameter in
        :py:meth:`update_lkf_enc_imu`.
        :return:
        """
        # Innovation (difference between measurement and our prediction)
        y = np.subtract(z, np.dot(self.H_enc, self.out_x))

        # Innovation uncertainty matrix
        self.S_pre = np.add(_multi_dot_three(self.H_enc, self.P, self.H_enc.T),
                            self.R_enc)
        # Kalman gain
        self.K_pre = _multi_dot_three(self.P, self.H_enc.T,
                                      np.linalg.inv(self.S_pre))

        np.subtract(self.I_6x6, np.dot(self.K_pre, self.H_enc), out=self.ikh)

        # Updated state (previous state + gain * innovation)
        self.out_x = np.add(self.out_x, np.dot(self.K_pre, y))

        _multi_dot_three(self.ikh, self.P, self.ikh.T, out=self.preal_a),
        _multi_dot_three(self.K_pre, self.R_enc, self.K_pre.T,
                         out=self.preal_b),

        # Updated state uncertainty
        self.P = np.add(self.preal_a, self.preal_b,
                        dtype=np.float32
                        )

    def _update_lkf_gps(self, z):
        """
        Update the filter with measurements from the GPS.

        :param z: Same as the `measurement` parameter in
        :py:meth:`update_lkf_gps`.
        :return:
        """
        y = np.subtract(z, np.dot(self.H_gps, self.out_x))
        # print(f"y {y}")
        # print(np.dot(self.H_gps, self.out_x))
        # Innovation uncertainty matrix
        self.S_pre = _multi_dot_three(self.H_gps, self.P,
                                      self.H_gps.T) + self.R_gps
        # Kalman gain
        self.K_pre = _multi_dot_three(self.P, self.H_gps.T,
                                      np.linalg.inv(self.S_pre))

        # Set Kalman gain to 0 for acceleration measurements
        self.K_pre[0, 0] = 0
        self.K_pre[3, 1] = 0
        # Precompute I - K * H
        np.subtract(self.I_6x6, np.dot(self.K_pre, self.H_gps), out=self.ikh)

        # Updated state (previous state + gain * innovation)
        self.out_x = np.add(self.out_x, np.dot(self.K_pre, y))

        _multi_dot_three(self.ikh, self.P, self.ikh.T, out=self.preal_a),
        _multi_dot_three(self.K_pre, self.R_gps, self.K_pre.T,
                         out=self.preal_b),

        # Updated state uncertainty
        self.P = np.add(
            self.preal_a,
            self.preal_b,
            dtype=np.float32
        )

    def set_K_pre_acc(self, value):
        self.K_pre[2, 0] = value
        self.K_pre[2, 1] = value
        self.K_pre[5, 0] = value
        self.K_pre[5, 1] = value


    def set_K_pre_vel(self, value):
        self.K_pre[1, 0] = value
        self.K_pre[4, 1] = value

    def _update_lkf_camera(self, z):
        """
        Update the filter with measurements from the camera.

        :param z: Same as the `measurement` parameter in
        :py:meth:`update_lkf_camera`.
        :return:
        """
        y = np.subtract(z, np.dot(self.H_gps, self.out_x))

        # Innovation uncertainty matrix
        self.S_pre = _multi_dot_three(self.H_gps, self.P,
                                      self.H_gps.T) + self.R_cam
        # Kalman gain
        self.K_pre = _multi_dot_three(self.P, self.H_gps.T,
                                      np.linalg.inv(self.S_pre))
        
        # Set Kalman gain to 0 for acceleration measurements
        self.set_K_pre_acc(0)
        self.set_K_pre_vel(0)
        # print(f"Camera Kalman gain {self.K_pre[0, 0]} {self.K_pre[3, 1]}") 
        # print(f"Camera Kalman gain \n {self.K_pre}")

        # Precompute I - K * H
        np.subtract(self.I_6x6, np.dot(self.K_pre, self.H_gps), out=self.ikh)

        # Updated state (previous state + gain * innovation)
        self.out_x = np.add(self.out_x, np.dot(self.K_pre, y))

        _multi_dot_three(self.ikh, self.P, self.ikh.T, out=self.preal_a),
        _multi_dot_three(self.K_pre, self.R_cam, self.K_pre.T,
                         out=self.preal_b),
        # Updated state uncertainty
        self.P = np.add(
            self.preal_a,
            self.preal_b,
            dtype=np.float32
        )

    
    def _get_predicted_state(self, dt):
        """
        Predict the state of the system after a given period of time.

        Updates the x (state) vector for a moment of time dt seconds 
        ahead of the current moment

        :param dt: Number of seconds 

        """

        t = dt
        # Frequent terms in matrices
        t2 = t ** 2

        # Transition matrix
        F = np.array(
            [[1,    t,  t2 / 2, 0,  0,  0],
             [0,    1,  t,      0,  0,  0],
             [0,    0,  1,      0,  0,  0],
             [0,    0,  0,      1,  t,  t2 / 2],
             [0,    0,  0,      0,  1,  t],
             [0,    0,  0,      0,  0,  1]],
            dtype=np.float32
        )

        # Calculate the new state of the system, based on the previous state.
        # x' = F * x
        out_x = np.matmul(F, self.out_x, dtype=np.float32)
        
        return out_x
    
    def get_predicted_pos(self, dt):
        out_x = self._get_predicted_state(dt)
        return out_x[0][0], out_x[3][0]
    
    def get_predicted_vel(self, dt):
        out_x = self._get_predicted_state(dt)
        return out_x[1][0], out_x[4][0]
    
    def get_predicted_acc(self, dt):
        out_x = self._get_predicted_state(dt)
        return out_x[2][0], out_x[5][0]

    def get_pos(self):
        return self.out_x[0][0], self.out_x[3][0]

    def get_vel(self):
        return self.out_x[1][0], self.out_x[4][0]

    def get_acc(self):
        return self.out_x[2][0], self.out_x[5][0]

    def get_px(self):
        return self.out_x[0][0]

    def get_py(self):
        return self.out_x[3][0]

    def get_vx(self):
        return self.out_x[1][0]

    def get_vy(self):
        return self.out_x[4][0]

    def get_ax(self):
        return self.out_x[2][0]

    def get_ay(self):
        return self.out_x[5][0]



class LinearKalmanFilterPosVel:
    """
    OUTDATED DESCRIPTION
    Models a Kalman filter that tracks 6 parameters (position, velocity,
    acceleration) on both x and y axes.
    OUTDATED DESCRIPTION

    .. TODO: The current implementation uses a 6 state system (px, vx, ax, py,
        vy, ay). Thus, we have a 6x6 uncertainty matrix (and other large
        matrices) that contribute to long matrix multiplication times.
        A better implementation would split the system in two 3-state
        subsystems, one containing position, velocity and acceleration
        values for the x axis, and one for the y axis.
    
    OUTDATED DESCRIPTION
    The different 'update' methods are called when sensor data is received.
    Each 'update' method internally calls the 'predict()' method first, which
    updates the state vector `x`. Then, the update step is run, which again
    updates state vector `x` (but this time with values from the measurements).
    After these two steps (predict & update), the x and y coordinates from the
    state vector are copied to the outptut vector, `x`.

                ┌───────────────────────┐
                │                       │
                │                       │
                │                       │
    measurement │      Linear           │       x
    ───────────►│--┐   Kalman       ┌---├────────►
                │  |   Filter       |   │
                │  |                |   │
                │  └---> update()---┘   │
                │     [&predict()]      │
                │                       │
                └───────────────────────┘

    NOTE: Each input/output has an "in"/"out" prefix. The comments will use
    `out_variable` and `variable` interchangeably to refer to the same output
    variable.
5 of size NxN, where N is the size of `out_x`
        The diagonal of this matrix gives a rough estimate of the probability
        of the corresponding states from `out_x`
    """

    def __init__(self, x_init=None, P_init=None):
        """
        Initialize the state and uncertainty matrices of the Kalman filter.
        OUTDATED DESCRIPTION

        :param x_init: Initial state vector (1 x 6 matrix)
        :param P_init: Initial uncertainty (6x6 matrix)
        """
        if x_init is not None:
            self.out_x = np.array(x_init, dtype=np.float32)
            assert self.out_x.size == 4, "Initial state vector must have 4 elements"
        else:
            self.out_x = np.array([[0, 0, 0, 0]], dtype=np.float32).T
        if P_init is not None:
            self.P = np.array(P_init, dtype=np.float32)
        else:
            self.P = np.array(
                [[0.1,  0,     0,   0,],
                 [0,    100,   0,   0,],
                 [0,    0,     0.1, 0,],
                 [0,    0,     0,   100,]], dtype=np.float32
            )

        # Squared acceleration noises in each direction
        self.ax2 = 1
        self.ay2 = 1
        # Identity matrix
        # Measurement uncertainty - encoder & IMU
        self.R_enc = np.array(
            [[1, 0],
             [0, 1]], dtype=np.float32
        )

        # Measurement uncertainty - GPS
        self.R_gps = np.array(
            [[1, 0],
             [0, 1]], dtype=np.float32
        )

        # Measurement uncertainty - camera
        self.R_cam = np.array(
            [[0.2, 0],
             [0, 0.2]], dtype=np.float32
        )

        # Measurement function - encoder
        self.H_enc = np.array(
            [[0, 1, 0, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )

        # Measurement function - GPS
        self.H_gps = np.array(
            [[1, 0, 0, 0],
             [0, 0, 1, 0]], dtype=np.float32
        )

        # Preallocate matrices for certain multiplication operations.
        self.S_pre = np.zeros((2, 2), dtype=np.float32)
        self.K_pre = np.zeros((4, 2), dtype=np.float32)
        self.ikh = np.zeros((4, 4), dtype=np.float32)
        self.preal_a = np.zeros((4, 4), dtype=np.float32)
        self.preal_b = np.zeros((4, 4), dtype=np.float32)
        self.I_4x4 = np.eye(len(self.out_x))

        # Flags
        self._is_initialized_enc = False
        self._is_initialized_gps_or_camera = False
        self._last_measurement_time = None

    def set_R_enc(self, R_enc):
        self.R_enc = R_enc

    def set_R_gps(self, R_gps):
        self.R_gps = R_gps

    def set_R_cam(self, R_cam):
        self.R_cam = R_cam

    def set_ax2(self, ax2):
        self.ax2 = ax2

    def set_ay2(self, ay2):
        self.ay2 = ay2
    
    def set_acc_noise(self, ax2, ay2):
        self.ax2 = ax2
        self.ay2 = ay2

    def update_enc_imu(self, measurement):
        """
        Update the filter with values coming from the encoder and IMU.
        OUTDATED DESCRIPTION

        This is only a wrapper method. Internally, it calls another method,
        ``_update_enc_imu``, that executes the update.

        :param measurement: Vector containing two values, vx and vy, giving
        the speed on each of the vx and vy axes.
        :return:
        """

        # If the filter was not initialized, initialize it with the first
        # velocity measurement.
        if not self._is_initialized_enc:
            self._last_measurement_time = time.time()
            self.out_x[1] = measurement[0]
            self.out_x[3] = measurement[1]
            self._is_initialized_enc = True
            return
        # Compute the time interval between last and current measurement
        dt = time.time() - self._last_measurement_time
        # Store the time of the current measurement
        self._last_measurement_time = time.time()
        # Predict, then update
        self._predict_lkf(dt)
        self._update_lkf_enc_imu(measurement)

    def update_gps(self, measurement):
        """
        Update the filter with values coming from the GPS.
        OUTDATED DESCRIPTION

        This is only a wrapper method. Internally, it calls another method,
        ``_update_gps``, that executes the update.

        :param measurement: Vector containing two values, px and py, giving
        the position on both x and y axes.
        :return:
        """
        if not self._is_initialized_gps_or_camera:
            # GPS and camera will have same init time, because they initialize the same
            # parameters (vx, vy) AND because we only do this initialization from one source
            # (either gps, or camera)
            self._last_measurement_time = time.time()
            self.out_x[0] = measurement[0]
            self.out_x[2] = measurement[1]
            self._is_initialized_gps_or_camera = True
            return
        # Compute the time interval between last and current measurement
        dt = time.time() - self._last_measurement_time
        # Store the time of the current measurement
        self._last_measurement_time = time.time()
        # Predict, then update
        self._predict_lkf(dt)
        self._update_lkf_gps(measurement)

    def update_camera(self, measurement):
        """
        Update the filter with measurements coming from the camera.
        OUTDATED DESCRIPTION

        This is only a wrapper method. Internally, it calls another method,
        ``_update_camera``, that executes the update.

        :param measurement: Vector containing two values, px and py, giving
        the position on both x and y axes.
        :return:
        """

        if not self._is_initialized_gps_or_camera:
            # GPS and camera will have the same init time, because they
            # initialize  the same parameters (vx, vy) AND because we only do
            # this initialization from one source (either gps, or camera).
            self._last_measurement_time = time.time()
            self.out_x[0] = measurement[0]
            self.out_x[3] = measurement[1]
            self._is_initialized_gps_or_camera = True
            return
        # Compute the time interval between last and current measurement
        dt = time.time() - self._last_measurement_time
        # Store the time of the current measurement
        self._last_measurement_time = time.time()
        # Predict, then update
        self._predict_lkf(dt)
        self._update_lkf_camera(measurement)

    def _predict_lkf(self, dt):
        """
        Predict the state of the system after a given period of time.
        OUTDATED DESCRIPTION

        Updates the x (state) vector, together with the uncertainity matrix
        (P) based on the time `dt` passed since the last measurement.

        :param dt: Time interval passed from the last measurement.
        :return: None
        """
        # dt == 0 means two measurements came with the same timestamp, one of them 
        # has already been used for updating the LKF, but now the second one
        # also tries to update the LKF.
        # It will make the predict() & update() calls, but in this case 
        # predict() isn't needed, since a previous measurement already 
        # trigerred a predict() step for the same timestamp.
        if dt == 0:
            return
        
        t = dt
        # Frequent terms in matrices
        t2 = t ** 2
        t3 = t ** 3
        t4 = t ** 4

        # Process noise matrix
        Q = np.array(
            [[t4 / 4 * self.ax2,    t3 / 2 * self.ax2,  0,                  0                   ],
             [t3 / 2 * self.ax2,    t2 * self.ax2,      0,                  0                   ],
             [0,                    0,                  t4 / 4 * self.ay2,  t3 / 2 * self.ay2,  ],
             [0,                    0,                  t3 / 2 * self.ay2,  t2 * self.ay2,      ]],
            dtype=np.float32) 
        # Transition matrix
        F = np.array(
            [[1,    t,  0,  0],
             [0,    1,  0,  0],
             [0,    0,  1,  t],
             [0,    0,  0,  1]],
            dtype=np.float32
        )

        # Calculate the new state of the system, based on the previous state.
        # x' = F * x
        self.out_x = np.matmul(F, self.out_x, dtype=np.float32)
        # Calculate the covariance matrix (uncertainty estimate)
        # P' = F * P * F.T + Q
        self.P = _multi_dot_three(F, self.P, F.T)
        self.P = self.P + Q

    def _update_lkf_enc_imu(self, z):
        """
        Update the filter with measurements from the encoder and the imu.

        :param z: Same as the `measurement` parameter in
        :py:meth:`update_lkf_enc_imu`.
        :return:
        """
        # Innovation (difference between measurement and our prediction)
        y = np.subtract(z, np.dot(self.H_enc, self.out_x))

        # Innovation uncertainty matrix
        self.S_pre = np.add(_multi_dot_three(self.H_enc, self.P, self.H_enc.T),
                            self.R_enc)
        # Kalman gain
        self.K_pre = _multi_dot_three(self.P, self.H_enc.T,
                                      np.linalg.inv(self.S_pre))

        np.subtract(self.I_4x4, np.dot(self.K_pre, self.H_enc), out=self.ikh)

        # Updated state (previous state + gain * innovation)
        self.out_x = np.add(self.out_x, np.dot(self.K_pre, y))

        _multi_dot_three(self.ikh, self.P, self.ikh.T, out=self.preal_a),
        _multi_dot_three(self.K_pre, self.R_enc, self.K_pre.T,
                         out=self.preal_b),

        # Updated state uncertainty
        self.P = np.add(self.preal_a, self.preal_b,
                        dtype=np.float32
                        )

    def _update_lkf_gps(self, z):
        """
        Update the filter with measurements from the GPS.

        :param z: Same as the `measurement` parameter in
        :py:meth:`update_lkf_gps`.
        :return:
        """
        y = np.subtract(z, np.dot(self.H_gps, self.out_x))

        # Innovation uncertainty matrix
        self.S_pre = _multi_dot_three(self.H_gps, self.P,
                                      self.H_gps.T) + self.R_gps
        # Kalman gain
        self.K_pre = _multi_dot_three(self.P, self.H_gps.T,
                                      np.linalg.inv(self.S_pre))

        # Precompute I - K * H
        np.subtract(self.I_4x4, np.dot(self.K_pre, self.H_gps), out=self.ikh)

        # Updated state (previous state + gain * innovation)
        self.out_x = np.add(self.out_x, np.dot(self.K_pre, y))

        _multi_dot_three(self.ikh, self.P, self.ikh.T, out=self.preal_a),
        _multi_dot_three(self.K_pre, self.R_gps, self.K_pre.T,
                         out=self.preal_b),

        # Updated state uncertainty
        self.P = np.add(
            self.preal_a,
            self.preal_b,
            dtype=np.float32
        )

    def _update_lkf_camera(self, z):
        """
        Update the filter with measurements from the camera.

        :param z: Same as the `measurement` parameter in
        :py:meth:`update_lkf_camera`.
        :return:
        """
        y = np.subtract(z, np.dot(self.H_gps, self.out_x))

        # Innovation uncertainty matrix
        self.S_pre = _multi_dot_three(self.H_gps, self.P,
                                      self.H_gps.T) + self.R_cam
        # Kalman gainQ
        self.K_pre = _multi_dot_three(self.P, self.H_gps.T,
                                      np.linalg.inv(self.S_pre))
        
        # Precompute I - K * H
        np.subtract(self.I_4x4, np.dot(self.K_pre, self.H_gps), out=self.ikh)

        # Updated state (previous state + gain * innovation)
        self.out_x = np.add(self.out_x, np.dot(self.K_pre, y))

        _multi_dot_three(self.ikh, self.P, self.ikh.T, out=self.preal_a),
        _multi_dot_three(self.K_pre, self.R_cam, self.K_pre.T,
                         out=self.preal_b),
        # Updated state uncertainty
        self.P = np.add(
            self.preal_a,
            self.preal_b,
            dtype=np.float32
        )

    def get_pos(self):
        return self.out_x[0][0], self.out_x[2][0]

    def get_vel(self):
        return self.out_x[1][0], self.out_x[3][0]

    def get_px(self):
        return self.out_x[0][0]

    def get_py(self):
        return self.out_x[2][0]

    def get_vx(self):
        return self.out_x[1][0]

    def get_vy(self):
        return self.out_x[3][0]



class StaticFilter():
    """
    This class implements a static filter (i.e. the estimated value is constant).
    It will be used mainly for position estimation when the car does not move.
    
    The filter basically averages all values received over time.
    """
    
    def __init__(self, state_init=None):
        """
        Initialize a static filter.

        :param state_init: _description_, defaults to None
        """
        if state_init is not None:
            self._state = state_init
        else:
            self._state = 0
        self._iteration = 1
        self._is_initialized = False

    def update(self, measurement):
        print("Static filter updated")
        if not self._is_initialized:
            self._state = measurement
            self._is_initialized = True
        self._state = self._state + 1/self._iteration * (measurement - self._state)
        self._iteration += 1

    def get_state(self):
        return self._state