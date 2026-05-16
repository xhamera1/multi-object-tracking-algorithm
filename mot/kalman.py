from __future__ import annotations

import numpy as np

from .types import BBox


class KalmanFilter:
    """Constant-velocity Kalman filter for `[x, y, w, h, vx, vy, vw, vh]`.

    Process and measurement noise are scaled by the current box height so the
    filter behaves consistently across object scales.
    """

    _std_weight_position = 1.0 / 20.0
    _std_weight_velocity = 1.0 / 160.0

    def __init__(self, bbox: BBox) -> None:
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        self.x = np.array([x, y, w, h, 0.0, 0.0, 0.0, 0.0], dtype=float)

        self.F = np.eye(8, dtype=float)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.F[3, 7] = 1.0

        self.H = np.zeros((4, 8), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
            ],
            dtype=float,
        )
        self.P = np.diag(std * std)

    def _process_noise(self) -> np.ndarray:
        h = max(1.0, float(self.x[3]))
        std_pos = self._std_weight_position * h
        std_vel = self._std_weight_velocity * h
        std = np.array(
            [std_pos, std_pos, std_pos, std_pos, std_vel, std_vel, std_vel, std_vel],
            dtype=float,
        )
        return np.diag(std * std)

    def _measurement_noise(self) -> np.ndarray:
        h = max(1.0, float(self.x[3]))
        std_pos = self._std_weight_position * h
        std = np.array([std_pos, std_pos, std_pos, std_pos], dtype=float)
        return np.diag(std * std)

    def predict(self) -> BBox:
        Q = self._process_noise()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return BBox(
            x=float(self.x[0]),
            y=float(self.x[1]),
            w=float(self.x[2]),
            h=float(self.x[3]),
        )

    def update(self, bbox: BBox) -> BBox:
        R = self._measurement_noise()

        z = np.array([[bbox.x], [bbox.y], [bbox.w], [bbox.h]], dtype=float)
        x_col = self.x.reshape(-1, 1)
        innovation = z - self.H @ x_col
        s = self.H @ self.P @ self.H.T + R
        ph_t = self.P @ self.H.T
        k = np.linalg.solve(s.T, ph_t.T).T
        x_new = x_col + k @ innovation
        self.x = x_new.flatten()
        self.P = (np.eye(8, dtype=float) - k @ self.H) @ self.P

        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])

        return BBox(
            x=float(self.x[0]),
            y=float(self.x[1]),
            w=float(self.x[2]),
            h=float(self.x[3]),
        )
