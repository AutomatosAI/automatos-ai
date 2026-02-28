"""
Attractor Dynamics for Agent Learning
=======================================

PRD-59 Phase 4: Evolves the simple EMA learning (Stage 6) into
attractor dynamics. Successful agent-task combinations become
attractors that pull future agent selection decisions.

Dynamics:
  dA/dt = α × (Q - Q_threshold) × A + η

Where:
  A             — attractor strength for (agent_id, task_type)
  Q             — observed quality score
  Q_threshold   — minimum acceptable quality (0.7)
  α             — learning rate (0.1)
  η ~ N(0, 0.01) — noise term (prevents local optima)

Storage: Redis hash for persistence across restarts.
Fallback: In-memory dict if Redis unavailable.
"""

import logging
import random
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AttractorLandscape:
    """
    Tracks attractor strengths for (agent_id, task_type) pairs.

    High-quality agent-task combinations strengthen their attractor,
    making the agent more likely to be selected for similar tasks.
    Low-quality combinations weaken, steering selection away.

    Usage:
        landscape = AttractorLandscape()
        landscape.update(agent_id=42, task_type="research", quality=0.92)
        strength = landscape.get_strength(agent_id=42, task_type="research")
    """

    DEFAULT_ALPHA = 0.1       # Learning rate
    DEFAULT_THRESHOLD = 0.7   # Quality threshold
    DEFAULT_NOISE_STD = 0.01  # Noise standard deviation
    DEFAULT_INITIAL = 0.5     # Initial attractor strength

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        alpha: float = DEFAULT_ALPHA,
        threshold: float = DEFAULT_THRESHOLD,
        noise_std: float = DEFAULT_NOISE_STD,
    ):
        """
        Args:
            redis_client: Optional Redis client for persistence.
            alpha: Learning rate.
            threshold: Quality threshold Q_threshold.
            noise_std: Standard deviation of noise term.
        """
        self.redis = redis_client
        self.alpha = alpha
        self.threshold = threshold
        self.noise_std = noise_std
        self._memory: Dict[str, float] = {}
        self._use_redis = redis_client is not None

    def _key(self, agent_id: int, task_type: str) -> str:
        return f"{agent_id}:{task_type}"

    def update(
        self,
        agent_id: int,
        task_type: str,
        quality: float,
    ) -> float:
        """
        Update attractor strength after an execution.

        dA/dt = α × (Q - Q_threshold) × A + η

        Args:
            agent_id: The agent that executed
            task_type: Type of task executed
            quality: Observed quality score (0-1)

        Returns:
            New attractor strength
        """
        key = self._key(agent_id, task_type)
        current = self.get_strength(agent_id, task_type)

        # Attractor dynamics
        delta = self.alpha * (quality - self.threshold) * current
        noise = random.gauss(0, self.noise_std)
        new_strength = max(0.0, min(1.0, current + delta + noise))

        self._set_strength(key, new_strength)

        logger.debug(
            f"Attractor update: agent={agent_id}, task={task_type}, "
            f"Q={quality:.2f}, A: {current:.3f} → {new_strength:.3f}"
        )

        return new_strength

    def get_strength(self, agent_id: int, task_type: str) -> float:
        """Get current attractor strength for an (agent, task_type) pair"""
        key = self._key(agent_id, task_type)

        if self._use_redis:
            try:
                val = self.redis.hget("neural_field:attractors", key)
                if val is not None:
                    return float(val)
            except Exception:
                pass

        return self._memory.get(key, self.DEFAULT_INITIAL)

    def get_all_strengths(self, agent_id: int) -> Dict[str, float]:
        """Get all attractor strengths for an agent across task types"""
        prefix = f"{agent_id}:"
        result = {}

        if self._use_redis:
            try:
                all_data = self.redis.hgetall("neural_field:attractors")
                for k, v in all_data.items():
                    k_str = k if isinstance(k, str) else k.decode()
                    if k_str.startswith(prefix):
                        task_type = k_str[len(prefix):]
                        result[task_type] = float(v)
                return result
            except Exception:
                pass

        for k, v in self._memory.items():
            if k.startswith(prefix):
                task_type = k[len(prefix):]
                result[task_type] = v

        return result

    def get_top_agents(
        self,
        task_type: str,
        top_k: int = 5,
    ) -> list[Tuple[int, float]]:
        """Get top agents for a task type by attractor strength"""
        results = []

        if self._use_redis:
            try:
                all_data = self.redis.hgetall("neural_field:attractors")
                for k, v in all_data.items():
                    k_str = k if isinstance(k, str) else k.decode()
                    if k_str.endswith(f":{task_type}"):
                        agent_id = int(k_str.split(":")[0])
                        results.append((agent_id, float(v)))
            except Exception:
                pass
        else:
            suffix = f":{task_type}"
            for k, v in self._memory.items():
                if k.endswith(suffix):
                    agent_id = int(k.split(":")[0])
                    results.append((agent_id, v))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _set_strength(self, key: str, value: float) -> None:
        self._memory[key] = value

        if self._use_redis:
            try:
                self.redis.hset("neural_field:attractors", key, str(value))
            except Exception as e:
                logger.debug(f"Redis attractor write failed: {e}")

    def reset(self, agent_id: Optional[int] = None) -> None:
        """Reset attractor strengths"""
        if agent_id is not None:
            prefix = f"{agent_id}:"
            keys_to_remove = [k for k in self._memory if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._memory[k]
        else:
            self._memory.clear()
