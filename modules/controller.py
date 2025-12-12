# modules/controller.py

"""
Alpha controller for adaptive risk in conformal prediction.

This module exposes an AlphaController class that adjusts the conformal
risk level alpha based on the current playback buffer, using a logistic
mapping:

    alpha = min_alpha + (max_alpha - min_alpha) / (
        1 + exp(steepness * (buffer - midpoint))
    )

- Large buffer  -> alpha ~ min_alpha (more conservative, larger radius)
- Small buffer  -> alpha ~ max_alpha (more aggressive, smaller radius)

An optional EMA smoothing factor controls how quickly alpha responds
to changes in the buffer.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class AlphaControllerConfig:
    """
    Configuration for AlphaController.

    Attributes:
        min_alpha: Lower bound on alpha (most conservative / safest).
        max_alpha: Upper bound on alpha (most aggressive / riskiest).
        max_buffer: Buffer level (in seconds) considered "fully healthy".
                    Any buffer >= max_buffer is treated as max_buffer.
        smooth_factor: EMA factor in (0, 1]. Smaller = slower, smoother
                       changes; larger = more reactive, jerkier.
    """
    min_alpha: float = 0.01
    max_alpha: float = 0.15
    midpoint: float = 13.0
    steepness: float = 0.15   
    smooth_factor: float = 1.0


class AlphaController:
    """
    Alpha controller that adapts conformal risk alpha based on buffer level.

    Typical usage:
        cfg = AlphaControllerConfig(
            min_alpha=0.01,
            max_alpha=0.20,
            max_buffer=5.0,
            smooth_factor=0.2,
        )
        controller = AlphaController(cfg)

        # each simulation / playback step:
        alpha = controller.update(current_buffer_seconds)

    The returned alpha can then be passed into your conformal radius
    computation, e.g.:
        radius = get_quantile_radius(residuals, alpha)
    """

    def __init__(self, config: Optional[AlphaControllerConfig] = None):
        self.cfg = config or AlphaControllerConfig()

        # Start at conservative (low risk) alpha
        self._alpha = self.cfg.min_alpha

    @property
    def alpha(self) -> float:
        """Current alpha value (after smoothing)."""
        return self._alpha

    def reset(self, alpha: Optional[float] = None) -> None:
        """
        Reset the controller's internal alpha.

        Args:
            alpha: Optional value to reset to. If None, resets to min_alpha.
        """
        if alpha is None:
            alpha = self.cfg.min_alpha
        self._alpha = float(alpha)

    def _buffer_to_target_alpha(self, buffer_level: float) -> float:
        """
        Map the current buffer level to an unsmoothed target alpha using
        a logistic function:

            target = min_alpha + (max_alpha - min_alpha) / (
                1 + exp(steepness * (buffer_level - midpoint))
            )
        """
        cfg = self.cfg

        # Logistic denominator
        x = float(buffer_level)

        denom = 1.0 + math.exp(cfg.steepness * (x - cfg.midpoint))

        frac = (cfg.max_alpha - cfg.min_alpha) / denom
        target_alpha = cfg.min_alpha + frac

        # Extra clamp in case of any numeric weirdness
        target_alpha = max(cfg.min_alpha, min(target_alpha, cfg.max_alpha))

        return target_alpha

    def update(self, current_buffer: float) -> float:
        """
        Update alpha given the current buffer level.

        Args:
            current_buffer: Current playback buffer in seconds.

        Returns:
            Updated alpha (float), after smoothing and clamping.
        """
        cfg = self.cfg

        # Step 1: compute the raw / ideal alpha for this buffer
        target_alpha = self._buffer_to_target_alpha(current_buffer)

        # Step 2: smooth it with an exponential moving average (EMA)
        s = cfg.smooth_factor
        # EMA: new_alpha = (1 - s) * old + s * target
        new_alpha = (1.0 - s) * self._alpha + s * target_alpha

        # Step 3: final clamp for safety (floating-point guard)
        new_alpha = max(cfg.min_alpha, min(new_alpha, cfg.max_alpha))

        self._alpha = new_alpha
        return self._alpha
