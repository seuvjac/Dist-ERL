"""Dynamic RL↔EA coupling policies (migration gating, sync helpers)."""

from typing import Tuple


class MigrationGate:
    """
    Early decoupling + performance-triggered RL→EA migration.

    - First ``warmup_frac`` of training: no migration (EA/RL explore independently).
    - After warmup: migrate only if RL eval (actor-aligned) beats EA median fitness
      for ``beats_required`` consecutive generations.
    """

    def __init__(self, max_generations: int, warmup_frac: float = 0.3,
                 beats_required: int = 3, margin: float = 0.05):
        self.warmup_end = max(1, int(max_generations * warmup_frac))
        self.beats_required = max(1, beats_required)
        self.margin = max(0.0, float(margin))
        self.beats_streak = 0

    def allow_migration(self, generation: int, eval_rl_aligned: float, ea_median_fitness: float) -> Tuple[bool, str]:
        if generation < self.warmup_end:
            self.beats_streak = 0
            return False, f"warmup({generation}/{self.warmup_end})"

        threshold = ea_median_fitness + self.margin * max(1.0, abs(ea_median_fitness))
        if eval_rl_aligned > threshold:
            self.beats_streak += 1
        else:
            self.beats_streak = 0

        if self.beats_streak >= self.beats_required:
            return True, f"rl_beats_ea({self.beats_streak}/{self.beats_required})"
        return False, f"gate({self.beats_streak}/{self.beats_required})"
