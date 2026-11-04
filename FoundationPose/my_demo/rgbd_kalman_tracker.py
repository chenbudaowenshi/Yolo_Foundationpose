#!/usr/bin/env python3
"""Small RGB-D 3D position tracker for the D435i pose pipeline.

The tracker deliberately estimates translation and velocity only. FoundationPose
provides rotation during global registration; the default real-time pipeline
then holds that rotation to avoid cylindrical-object drift. A constant-velocity
Kalman model smooths noisy mask/depth measurements, predicts through short
dropouts, rejects implausible jumps with a Mahalanobis gate, and reports when
the target has remained nearly stationary long enough for expensive 6D
registration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    acceleration_std_mps2: float = 0.10
    initial_position_std_m: float = 0.04
    initial_velocity_std_mps: float = 0.40
    measurement_xy_std_m: float = 0.008
    measurement_z_std_m: float = 0.015
    mahalanobis_gate: float = 16.27
    max_missed_frames: int = 8
    stationary_speed_mps: float = 0.030
    stationary_frames: int = 5
    maximum_dt_s: float = 0.25

    def validate(self) -> None:
        positive = {
            "acceleration_std_mps2": self.acceleration_std_mps2,
            "initial_position_std_m": self.initial_position_std_m,
            "initial_velocity_std_mps": self.initial_velocity_std_mps,
            "measurement_xy_std_m": self.measurement_xy_std_m,
            "measurement_z_std_m": self.measurement_z_std_m,
            "mahalanobis_gate": self.mahalanobis_gate,
            "stationary_speed_mps": self.stationary_speed_mps,
            "maximum_dt_s": self.maximum_dt_s,
        }
        invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"tracker values must be positive and finite: {', '.join(invalid)}")
        if self.max_missed_frames < 0 or self.stationary_frames <= 0:
            raise ValueError("max_missed_frames must be nonnegative and stationary_frames must be positive")


@dataclass(frozen=True)
class TrackerSnapshot:
    status: str
    position_m: np.ndarray | None
    velocity_mps: np.ndarray | None
    position_std_m: np.ndarray | None
    measurement_accepted: bool | None
    mahalanobis_distance: float | None
    age_frames: int
    hit_frames: int
    missed_frames: int
    stationary_frames: int
    is_stationary: bool

    def metadata(self) -> dict[str, str | bool | int | float | list[float] | None]:
        return {
            "status": self.status,
            "position_m": None if self.position_m is None else [float(value) for value in self.position_m],
            "velocity_mps": None if self.velocity_mps is None else [float(value) for value in self.velocity_mps],
            "speed_mps": None if self.velocity_mps is None else float(np.linalg.norm(self.velocity_mps)),
            "position_std_m": (
                None if self.position_std_m is None else [float(value) for value in self.position_std_m]
            ),
            "measurement_accepted": self.measurement_accepted,
            "mahalanobis_distance": self.mahalanobis_distance,
            "age_frames": self.age_frames,
            "hit_frames": self.hit_frames,
            "missed_frames": self.missed_frames,
            "stationary_frames": self.stationary_frames,
            "is_stationary": self.is_stationary,
        }


class ConstantVelocity3DTracker:
    """Six-state Kalman filter: ``[x, y, z, vx, vy, vz]``."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.config.validate()
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._last_timestamp_s: float | None = None
        self._age_frames = 0
        self._hit_frames = 0
        self._missed_frames = 0
        self._stationary_frames = 0
        self._last_measurement_accepted: bool | None = None
        self._last_mahalanobis: float | None = None

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def lost(self) -> bool:
        return self.initialized and self._missed_frames > self.config.max_missed_frames

    @property
    def position_m(self) -> np.ndarray | None:
        return None if self._state is None else self._state[:3].copy()

    @property
    def velocity_mps(self) -> np.ndarray | None:
        return None if self._state is None else self._state[3:].copy()

    @property
    def is_stationary(self) -> bool:
        return self.initialized and self._stationary_frames >= self.config.stationary_frames

    def reset(self) -> None:
        self._state = None
        self._covariance = None
        self._last_timestamp_s = None
        self._age_frames = 0
        self._hit_frames = 0
        self._missed_frames = 0
        self._stationary_frames = 0
        self._last_measurement_accepted = None
        self._last_mahalanobis = None

    def _initialize(
        self,
        measurement_m: np.ndarray,
        timestamp_s: float,
        measurement_covariance: np.ndarray,
    ) -> None:
        self._state = np.zeros(6, dtype=np.float64)
        self._state[:3] = measurement_m
        self._covariance = np.zeros((6, 6), dtype=np.float64)
        position_variance = np.maximum(
            np.diag(measurement_covariance),
            self.config.initial_position_std_m**2,
        )
        self._covariance[:3, :3] = np.diag(position_variance)
        self._covariance[3:, 3:] = np.eye(3) * self.config.initial_velocity_std_mps**2
        self._last_timestamp_s = timestamp_s
        self._age_frames = 1
        self._hit_frames = 1
        self._missed_frames = 0
        self._stationary_frames = 1
        self._last_measurement_accepted = True
        self._last_mahalanobis = 0.0

    def _predict(self, timestamp_s: float) -> None:
        assert self._state is not None and self._covariance is not None
        if self._last_timestamp_s is None:
            self._last_timestamp_s = timestamp_s
            return
        raw_dt = max(1e-4, timestamp_s - self._last_timestamp_s)
        dt = min(raw_dt, self.config.maximum_dt_s)
        self._last_timestamp_s = timestamp_s

        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3) * dt
        acceleration_variance = self.config.acceleration_std_mps2**2
        process_noise = np.block(
            [
                [np.eye(3) * dt**4 / 4.0, np.eye(3) * dt**3 / 2.0],
                [np.eye(3) * dt**3 / 2.0, np.eye(3) * dt**2],
            ]
        ) * acceleration_variance
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process_noise
        self._covariance = (self._covariance + self._covariance.T) * 0.5
        self._age_frames += 1

    def _default_measurement_covariance(self) -> np.ndarray:
        return np.diag(
            [
                self.config.measurement_xy_std_m**2,
                self.config.measurement_xy_std_m**2,
                self.config.measurement_z_std_m**2,
            ]
        )

    @staticmethod
    def _validate_measurement(measurement_m: np.ndarray | None) -> np.ndarray | None:
        if measurement_m is None:
            return None
        measurement = np.asarray(measurement_m, dtype=np.float64).reshape(-1)
        if measurement.shape != (3,) or not np.all(np.isfinite(measurement)):
            raise ValueError("3D tracker measurement must contain three finite values")
        return measurement

    @staticmethod
    def _validate_covariance(covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("measurement covariance must be a finite 3x3 matrix")
        if np.any(np.diag(covariance) <= 0):
            raise ValueError("measurement covariance diagonal must be positive")
        return (covariance + covariance.T) * 0.5

    def step(
        self,
        measurement_m: np.ndarray | None,
        timestamp_s: float,
        measurement_covariance: np.ndarray | None = None,
    ) -> TrackerSnapshot:
        """Predict to ``timestamp_s`` and optionally fuse one 3D measurement."""
        if not np.isfinite(timestamp_s):
            raise ValueError("tracker timestamp must be finite")
        measurement = self._validate_measurement(measurement_m)
        covariance = self._validate_covariance(
            self._default_measurement_covariance()
            if measurement_covariance is None
            else measurement_covariance
        )

        if not self.initialized:
            if measurement is not None:
                self._initialize(measurement, float(timestamp_s), covariance)
            return self.snapshot()

        if self.lost and measurement is not None:
            self._initialize(measurement, float(timestamp_s), covariance)
            return self.snapshot()

        self._predict(float(timestamp_s))
        self._last_measurement_accepted = None
        self._last_mahalanobis = None
        if measurement is None:
            self._missed_frames += 1
            self._stationary_frames = 0
            return self.snapshot()

        assert self._state is not None and self._covariance is not None
        observation_matrix = np.zeros((3, 6), dtype=np.float64)
        observation_matrix[:, :3] = np.eye(3)
        residual = measurement - observation_matrix @ self._state
        residual_covariance = observation_matrix @ self._covariance @ observation_matrix.T + covariance
        solved_residual = np.linalg.solve(residual_covariance, residual)
        mahalanobis = float(residual.T @ solved_residual)
        self._last_mahalanobis = mahalanobis

        if mahalanobis > self.config.mahalanobis_gate:
            self._last_measurement_accepted = False
            self._missed_frames += 1
            self._stationary_frames = 0
            return self.snapshot()

        gain = np.linalg.solve(
            residual_covariance,
            observation_matrix @ self._covariance,
        ).T
        self._state = self._state + gain @ residual
        identity = np.eye(6, dtype=np.float64)
        correction = identity - gain @ observation_matrix
        self._covariance = (
            correction @ self._covariance @ correction.T + gain @ covariance @ gain.T
        )
        self._covariance = (self._covariance + self._covariance.T) * 0.5
        self._last_measurement_accepted = True
        self._hit_frames += 1
        self._missed_frames = 0
        speed = float(np.linalg.norm(self._state[3:]))
        self._stationary_frames = self._stationary_frames + 1 if speed <= self.config.stationary_speed_mps else 0
        return self.snapshot()

    def snapshot(self) -> TrackerSnapshot:
        if self._state is None or self._covariance is None:
            return TrackerSnapshot(
                status="uninitialized",
                position_m=None,
                velocity_mps=None,
                position_std_m=None,
                measurement_accepted=self._last_measurement_accepted,
                mahalanobis_distance=self._last_mahalanobis,
                age_frames=self._age_frames,
                hit_frames=self._hit_frames,
                missed_frames=self._missed_frames,
                stationary_frames=self._stationary_frames,
                is_stationary=False,
            )
        if self.lost:
            status = "lost"
        elif self._missed_frames > 0:
            status = "coasting"
        else:
            status = "tracking"
        return TrackerSnapshot(
            status=status,
            position_m=self._state[:3].copy(),
            velocity_mps=self._state[3:].copy(),
            position_std_m=np.sqrt(np.maximum(0.0, np.diag(self._covariance)[:3])),
            measurement_accepted=self._last_measurement_accepted,
            mahalanobis_distance=self._last_mahalanobis,
            age_frames=self._age_frames,
            hit_frames=self._hit_frames,
            missed_frames=self._missed_frames,
            stationary_frames=self._stationary_frames,
            is_stationary=self.is_stationary,
        )
