import numpy as np
import scipy.linalg

# Table for the 0.95 quantile of the chi-square distribution with N degrees of freedom.
# This is used as a gating threshold in the Mahalanobis distance calculation.
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space,
    improved for tracking humans in a medium-sized retail clothing store.

    The 8-dimensional state space is:
    
        x, y, a, h, vx, vy, va, vh

    where (x, y) is the center of the bounding box, a is the aspect ratio,
    h is the height, and the remaining components are the respective velocities.
    
    Object motion is modeled using a constant velocity model. The bounding box
    measurement (x, y, a, h) is assumed to be a linear observation of the state.

    The filter parameters are tunable to adapt to the slower and more variable
    motion of humans, and to account for typical uncertainties encountered in a 
    retail environment.
    """

    def __init__(self, dt=1.0, std_weight_position=1./20, std_weight_velocity=1./160):
        """
        Initialize the Kalman filter.

        Parameters
        ----------
        dt : float, optional
            Time step between consecutive measurements. Adjust this based on
            the camera frame rate (e.g. dt=1/30 for 30 fps). Default is 1.0.
        std_weight_position : float, optional
            Standard deviation weight for the position components (x, y, a, h).
            This parameter can be tuned to reflect the measurement noise observed
            in human detection. Default is 1/20.
        std_weight_velocity : float, optional
            Standard deviation weight for the velocity components.
            Adjust this based on the expected human motion dynamics.
            Default is 1/160.
        """
        ndim = 4
        self.dt = dt
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # Create motion model matrix (8x8). The upper-right block links
        # position and velocity through the time step dt.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix (4x8) to extract the bounding box measurement.
        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement):
        """
        Create a new track from an initial bounding box measurement.

        Parameters
        ----------
        measurement : ndarray
            A 4-dimensional array representing the bounding box in the form
            (x, y, a, h), where (x, y) is the center, a is the aspect ratio,
            and h is the height.

        Returns
        -------
        mean : ndarray
            An 8-dimensional mean state vector, with velocities initialized to 0.
        covariance : ndarray
            An 8x8 covariance matrix representing the uncertainty in the state.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Initialize covariance with higher uncertainty in velocity.
        std = [
            2 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[3],  # y
            1e-2,                                            # a
            2 * self._std_weight_position * measurement[3],  # h
            10 * self._std_weight_velocity * measurement[3], # vx
            10 * self._std_weight_velocity * measurement[3], # vy
            1e-5,                                            # va
            10 * self._std_weight_velocity * measurement[3]  # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run the prediction step of the Kalman filter.

        Parameters
        ----------
        mean : ndarray
            The previous state mean (8-dimensional).
        covariance : ndarray
            The previous state covariance (8x8).

        Returns
        -------
        mean : ndarray
            The predicted state mean.
        covariance : ndarray
            The predicted state covariance, increased by the process noise.
        """
        # Process noise standard deviations for position and velocity.
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        # Motion covariance accounts for uncertainty in the constant velocity model.
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project the state distribution into the measurement space.

        Parameters
        ----------
        mean : ndarray
            The 8-dimensional state mean.
        covariance : ndarray
            The 8x8 state covariance.

        Returns
        -------
        mean_proj : ndarray
            The projected mean in measurement space (4-dimensional).
        covariance_proj : ndarray
            The projected covariance in measurement space.
        """
        # Innovation noise standard deviations.
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,  # slightly higher uncertainty for aspect ratio
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean_proj = np.dot(self._update_mat, mean)
        covariance_proj = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean_proj, covariance_proj + innovation_cov

    def update(self, mean, covariance, measurement):
        """
        Run the update (correction) step of the Kalman filter.

        Parameters
        ----------
        mean : ndarray
            The predicted state mean (8-dimensional).
        covariance : ndarray
            The predicted state covariance (8x8).
        measurement : ndarray
            The new measurement (4-dimensional) in the form (x, y, a, h).

        Returns
        -------
        new_mean : ndarray
            The updated state mean.
        new_covariance : ndarray
            The updated state covariance.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Use Cholesky factorization for numerical stability.
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                               np.dot(covariance, self._update_mat.T).T,
                                               check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Compute the squared Mahalanobis distance between the state distribution and measurements.

        A gating distance threshold (e.g. from chi2inv95) can be used to discard unlikely associations.
        If only_position is True, only the (x, y) position is considered.

        Parameters
        ----------
        mean : ndarray
            The 8-dimensional state mean.
        covariance : ndarray
            The 8x8 state covariance.
        measurements : ndarray
            An Nx4 array of measurements, each as (x, y, a, h).
        only_position : bool, optional
            If True, only consider the (x, y) positions. Default is False.

        Returns
        -------
        squared_maha : ndarray
            A 1D array of squared Mahalanobis distances for each measurement.
        """
        mean_proj, cov_proj = self.project(mean, covariance)
        if only_position:
            mean_proj, cov_proj = mean_proj[:2], cov_proj[:2, :2]
            measurements = measurements[:, :2]

        # Compute Cholesky factor of the covariance matrix.
        cholesky_factor = np.linalg.cholesky(cov_proj)
        d = measurements - mean_proj
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

    def gating_mask(self, mean, covariance, measurements, threshold, only_position=False):
        """
        Utility method to return a boolean mask for measurements that are within a given gating threshold.

        Parameters
        ----------
        mean : ndarray
            The 8-dimensional state mean.
        covariance : ndarray
            The 8x8 state covariance.
        measurements : ndarray
            An Nx4 array of measurements.
        threshold : float
            The squared Mahalanobis distance threshold for gating.
        only_position : bool, optional
            If True, only consider the position (x, y) for gating. Default is False.

        Returns
        -------
        mask : ndarray
            A boolean array of length N, where True indicates that the corresponding
            measurement is within the gating threshold.
        """
        distances = self.gating_distance(mean, covariance, measurements, only_position)
        return distances < threshold
