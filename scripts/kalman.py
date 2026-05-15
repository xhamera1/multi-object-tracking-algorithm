import numpy as np

from .iou import BBox


class Kalman:
    def __init__(self, initial_bbox: BBox) -> None:
        x, y, w, h = initial_bbox
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

        self.P = np.eye(8, dtype=float) * 10.0
        self.Q = np.eye(8, dtype=float)
        self.Q[:4, :4] *= 0.05
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4, dtype=float) * 1.0

    def predict(self) -> BBox:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3]))

    def update(self, bbox: BBox) -> None:
        z = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], dtype=float)
        x_col = self.x.reshape(-1, 1)
        y = z - self.H @ x_col
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        x_new = x_col + k @ y
        self.x = x_new.flatten()
        self.P = (np.eye(8, dtype=float) - k @ self.H) @ self.P

        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
