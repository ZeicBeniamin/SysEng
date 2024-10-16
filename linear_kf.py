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


class LinearKalmanFilterPosVel:
    """
    OUTDATED DESCRIPTION
    Models a Kalman filter that tracks 4 parameters (position, velocity,
    acceleration) on both x and y axes.
    OUTDATED DESCRIPTION

    .. TODO: The current implementation uses a 4 state system (px, vx, py,
        vy). Thus, we have a 4x4 uncertainty matrix (and other large
        matrices) that contribute to long matrix multiplication times.
        A better implementation would split the system in two 2-state
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
        The vector contains 4 states: [px, vx, py, vy]
    `out_P`: Covariance matrix of size NxN, where N is the size of `out_x`
        The diagonal of this matrix gives a rough estimate of the probability
        of the corresponding states from `out_x`
    """

    def __init__(self, x_init=None, P_init=None):
        """
        Initialize the state and uncertainty matrices of the Kalman filter.

        :param x_init: Initial state vector (1 x 4 matrix)
        :param P_init: Initial uncertainty (4x4 matrix)
        """
        # TODO: 4x1, shrink to 2x1
        if x_init is not None:
            self.out_x = np.array(x_init, dtype=np.float32)
            assert self.out_x.size == 2, "Initial state vector must have shape 2x1"
        else:
            self.out_x = np.array([[0, 0]], dtype=np.float32).T
        if P_init is not None:
            self.P = np.array(P_init, dtype=np.float32)
            assert self.P.shape == (2,2), "Initial P matrix must have shape 2x2"
        else:
            # TODO: 4x4, shrink to 2x2
            self.P = np.array(
                [[0.1,  0,  ],
                 [0,    100,]], dtype=np.float32
            )

        # Squared acceleration noises in each direction
        self.ax2 = 1e-5
        # self.ay2 = 1
        # Identity matrix
        # Measurement uncertainty - encoder & IMU
        # TODO: 2x2 - previously used for vx and vy, now used for px, vx
        self.R_enc = np.array(
            [[1e-5, 0],
             [0, 1e-5]], dtype=np.float32
        )

        # # Measurement uncertainty - GPS
        # self.R_gps = np.array(
        #     [[1, 0],
        #      [0, 1]], dtype=np.float32
        # )

        # Measurement function - encoder & imu
        # TODO: 2x4 used to update vx and vy in state vector x
                # now used to update px and vx, so needs resizing and reworking
        self.H_enc = np.array(
            [[1,  0],
             [0,  1]], dtype=np.float32
        )

        # # Measurement function - GPS
        # self.H_gps = np.array(
        #     [[1, 0, 0, 0],
        #      [0, 0, 1, 0]], dtype=np.float32
        # )

        # Preallocate matrices for certain multiplication operations.

        # TODO: Change sizes accordingly
        self.S_pre = np.zeros((2, 2), dtype=np.float32)
        # 4x2, might shrink
        self.K_pre = np.zeros((4, 2), dtype=np.float32)
        # 4x4, might shrink
        self.ikh = np.zeros((2, 2), dtype=np.float32)
        # TODO: 4x4, might shrink to 2x2
        self.preal_a = np.zeros((2, 2), dtype=np.float32)
        # TODO: 4x4, might shrink to 2x2
        self.preal_b = np.zeros((2, 2), dtype=np.float32)
        # TODO: 4x4, shrink to 2x2
        self.I_2x2 = np.eye(len(self.out_x))

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

    # def set_ay2(self, ay2):
    #     self.ay2 = ay2
    
    # def set_acc_noise(self, ax2, ay2):
    #     self.ax2 = ax2
    #     self.ay2 = ay2

    def update_enc_imu(self, measurement, time):
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
            self._last_measurement_time = time
            self.out_x[0] = measurement[0]
            self.out_x[1] = measurement[1]
            self._is_initialized_enc = True
            return
        # Compute the time interval between last and current measurement
        dt = time - self._last_measurement_time
        # Store the time of the current measurement
        self._last_measurement_time = time
        # Predict, then update
        self._predict_lkf(dt)
        self._update_lkf_enc_imu(measurement)

    # def update_gps(self, measurement):
    #     """
    #     Update the filter with values coming from the GPS.
    #     OUTDATED DESCRIPTION

    #     This is only a wrapper method. Internally, it calls another method,
    #     ``_update_gps``, that executes the update.

    #     :param measurement: Vector containing two values, px and py, giving
    #     the position on both x and y axes.
    #     :return:
    #     """
    #     if not self._is_initialized_gps_or_camera:
    #         # GPS and camera will have same init time, because they initialize the same
    #         # parameters (vx, vy) AND because we only do this initialization from one source
    #         # (either gps, or camera)
    #         self._last_measurement_time = time.time()
    #         self.out_x[0] = measurement[0]
    #         self.out_x[2] = measurement[1]
    #         self._is_initialized_gps_or_camera = True
    #         return
    #     # Compute the time interval between last and current measurement
    #     dt = time.time() - self._last_measurement_time
    #     # Store the time of the current measurement
    #     self._last_measurement_time = time.time()
    #     # Predict, then update
    #     self._predict_lkf(dt)
    #     self._update_lkf_gps(measurement)

    # def update_camera(self, measurement):
    #     """
    #     Update the filter with measurements coming from the camera.
    #     OUTDATED DESCRIPTION

    #     This is only a wrapper method. Internally, it calls another method,
    #     ``_update_camera``, that executes the update.

    #     :param measurement: Vector containing two values, px and py, giving
    #     the position on both x and y axes.
    #     :return:
    #     """

    #     if not self._is_initialized_gps_or_camera:
    #         # GPS and camera will have the same init time, because they
    #         # initialize  the same parameters (vx, vy) AND because we only do
    #         # this initialization from one source (either gps, or camera).
    #         self._last_measurement_time = time.time()
    #         self.out_x[0] = measurement[0]
    #         self.out_x[3] = measurement[1]
    #         self._is_initialized_gps_or_camera = True
    #         return
    #     # Compute the time interval between last and current measurement
    #     dt = time.time() - self._last_measurement_time
    #     # Store the time of the current measurement
    #     self._last_measurement_time = time.time()
    #     # Predict, then update
    #     self._predict_lkf(dt)
    #     self._update_lkf_camera(measurement)

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
        # TODO: 4x4 - shrink to 2x2
        Q = np.array(
            [[t4 / 4 * self.ax2,    t3 / 2 * self.ax2,],
             [t3 / 2 * self.ax2,    t2 * self.ax2,    ]],
            dtype=np.float32)
        
        # Transition matrix
        # TODO: 4x4 - shrink to 2x2
        F = np.array(
            [[1,    t,],
             [0,    1,]],
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
        # Same size as z - 2x1?
        y = np.subtract(z, np.dot(self.H_enc, self.out_x))

        # Innovation uncertainty matrix
        # 2x2 TODO: Check, but it may stay 2x2
        self.S_pre = np.add(_multi_dot_three(self.H_enc, self.P, self.H_enc.T),
                            self.R_enc)
        # Kalman gain
        # 2x2 TODO: Check, but stays
        self.K_pre = _multi_dot_three(self.P, self.H_enc.T,
                                      np.linalg.inv(self.S_pre))

        np.subtract(self.I_2x2, np.dot(self.K_pre, self.H_enc), out=self.ikh)

        # Updated state (previous state + gain * innovation)
        self.out_x = np.add(self.out_x, np.dot(self.K_pre, y))

        _multi_dot_three(self.ikh, self.P, self.ikh.T, out=self.preal_a),
        _multi_dot_three(self.K_pre, self.R_enc, self.K_pre.T,
                         out=self.preal_b),

        # Updated state uncertainty
        self.P = np.add(self.preal_a, self.preal_b,
                        dtype=np.float32
                        )

    # def _update_lkf_gps(self, z):
    #     """
    #     Update the filter with measurements from the GPS.

    #     :param z: Same as the `measurement` parameter in
    #     :py:meth:`update_lkf_gps`.
    #     :return:
    #     """
    #     y = np.subtract(z, np.dot(self.H_gps, self.out_x))

    #     # Innovation uncertainty matrix
    #     self.S_pre = _multi_dot_three(self.H_gps, self.P,
    #                                   self.H_gps.T) + self.R_gps
    #     # Kalman gain
    #     self.K_pre = _multi_dot_three(self.P, self.H_gps.T,
    #                                   np.linalg.inv(self.S_pre))

    #     # Precompute I - K * H
    #     np.subtract(self.I_4x4, np.dot(self.K_pre, self.H_gps), out=self.ikh)

    #     # Updated state (previous state + gain * innovation)
    #     self.out_x = np.add(self.out_x, np.dot(self.K_pre, y))

    #     _multi_dot_three(self.ikh, self.P, self.ikh.T, out=self.preal_a),
    #     _multi_dot_three(self.K_pre, self.R_gps, self.K_pre.T,
    #                      out=self.preal_b),

    #     # Updated state uncertainty
    #     self.P = np.add(
    #         self.preal_a,
    #         self.preal_b,
    #         dtype=np.float32
    #     )

    # def _update_lkf_camera(self, z):
    #     """
    #     Update the filter with measurements from the camera.

    #     :param z: Same as the `measurement` parameter in
    #     :py:meth:`update_lkf_camera`.
    #     :return:
    #     """
    #     y = np.subtract(z, np.dot(self.H_gps, self.out_x))

    #     # Innovation uncertainty matrix
    #     self.S_pre = _multi_dot_three(self.H_gps, self.P,
    #                                   self.H_gps.T) + self.R_cam
    #     # Kalman gainQ
    #     self.K_pre = _multi_dot_three(self.P, self.H_gps.T,
    #                                   np.linalg.inv(self.S_pre))
        
    #     # Precompute I - K * H
    #     np.subtract(self.I_4x4, np.dot(self.K_pre, self.H_gps), out=self.ikh)

    #     # Updated state (previous state + gain * innovation)
    #     self.out_x = np.add(self.out_x, np.dot(self.K_pre, y))

    #     _multi_dot_three(self.ikh, self.P, self.ikh.T, out=self.preal_a),
    #     _multi_dot_three(self.K_pre, self.R_cam, self.K_pre.T,
    #                      out=self.preal_b),
    #     # Updated state uncertainty
    #     self.P = np.add(
    #         self.preal_a,
    #         self.preal_b,
    #         dtype=np.float32
    #     )

    def get_pos(self):
        return self.out_x[0][0]

    def get_vel(self):
        return self.out_x[1][0]
    
    def get_state(self):
        return self.out_x