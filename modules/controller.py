# modules/controller.py

"""
Alpha controller for adaptive risk in conformal prediction.
This module exposes an AlphaController class that adjusts the conformal
risk level alpha based on the current playback buffer.
Concept:
    - alpha controls the risk / coverage tradeoff for conformal prediction.
    - Smaller alpha  -> higher coverage (bigger safety radius, more tiles).
    - Larger alpha   -> lower coverage (smaller safety radius, fewer tiles).
Policy (high-level):
    - When the playback buffer is healthy (large), we can afford to be safe:
        -> keep alpha near min_alpha (e.g., 0.01 => ~99% coverage).
    - When the buffer is shrinking / low, we must be more aggressive:
        -> increase alpha toward max_alpha (e.g., 0.20 => ~80% coverage).
Implementation details:
    - We linearly map current_buffer ∈ [0, max_buffer] to
      target_alpha ∈ [max_alpha, min_alpha], so:
        * buffer >= max_buffer -> alpha ≈ min_alpha
        * buffer  <= 0         -> alpha ≈ max_alpha
        * in between, alpha is interpolated.
    - We then apply an exponential moving average (EMA) between the
      previous alpha and the new target_alpha to avoid jerky jumps.
"""

from dataclasses import dataclass
from typing import Optional


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
    max_alpha: float = 0.20
    max_buffer: float = 5.0
    smooth_factor: float = 0.2  # 0.1–0.3 is usually reasonable


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

    def _buffer_to_target_alpha(self, current_buffer: float) -> float:
        """
        Map the current buffer level to an unsmoothed target alpha.
        We clamp buffer into [0, max_buffer] and then linearly map:
            buffer >= max_buffer  -> alpha = min_alpha   (very safe)
            buffer  <= 0          -> alpha = max_alpha   (very risky)
            in between            -> interpolate
        This matches the idea:
            target_alpha = min_alpha + (max_buffer - buffer) * slope
        but normalized so that the slope is (max_alpha - min_alpha) / max_buffer.
        """
        cfg = self.cfg
        buf = max(0.0, min(current_buffer, cfg.max_buffer))

        # fraction of how "low" the buffer is (0 = full, 1 = empty)
        low_frac = (cfg.max_buffer - buf) / cfg.max_buffer
        # interpolate between min_alpha (full buffer) and max_alpha (empty)
        target_alpha = cfg.min_alpha + low_frac * (cfg.max_alpha - cfg.min_alpha)

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