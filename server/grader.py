from dataclasses import dataclass, field
from typing import Dict

from engine.human_model import HumanState, Persona


@dataclass
class GradeResult:
    reward: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""


@dataclass
class ScenarioGradingConfig:
    task_name: str = "medium"
    max_total_reward: float = 1.0


def _normalize_task_name(task_name: str) -> str:
    aliases = {
        "easy": "easy-survival",
        "medium": "medium-eudaimonia",
        "hard": "hard-detox",
    }
    return aliases.get(task_name, task_name)


def _clamp_strict(val: float) -> float:
    """Ensures score is strictly between 0 and 1 (non-inclusive) as required by OpenEnv."""
    return max(0.01, min(0.99, val))


class Grader:
    """Evaluates the agent's performance based on the specific task objective."""

    def __init__(self, config: ScenarioGradingConfig | None = None):
        self._config = config or ScenarioGradingConfig()
        self._cumulative_reward = 0.0
        self._step_rewards = []
    
    @staticmethod
    def get_reward(task_name: str, state: HumanState, cortisol: float, done: bool) -> float:
        """
        Calculates meaningful reward based on task type.
        Supports partial progress.
        """
        task_name = _normalize_task_name(task_name)

        # Base penalty for high cortisol (normalized bounds)
        if cortisol > 0.75: # burnout threshold mapped to normalized 0.75
            return 0.0 # No negative rewards!
            
        base_reward = 0.0
        
        if task_name == "easy-survival":
            # Reward is 0.05 minus a tiny cortisol penalty. Max is 0.05.
            base_reward = max(0.0, 0.05 - (cortisol * 0.05))
            
        elif task_name == "medium-eudaimonia":
            # Maximize wisdom (driven by competence and relatedness). Max 0.05.
            base_reward = max(0.0, 0.05 * ((state.competence + state.relatedness) / 2.0))
            
        elif task_name == "hard-detox":
            # Specifically recover autonomy and relatedness.
            base_reward = max(0.0, 0.05 * ((state.autonomy + state.relatedness) / 2.0))
            
        import statistics
        
        # Penalty for Over-Exploitation (repetitive algorithm loophole)
        if state.autonomy < 0.3:
            base_reward *= 0.5
            
        # Volatility / Flow State Logic
        if len(state.dopamine_history) > 2:
            volatility = statistics.stdev(state.dopamine_history)
            if volatility > 0.20:
                base_reward *= 0.5 # Penalty for Rollercoaster dopamine-farming
            elif volatility < 0.10 and state.dopamine > 0.5:
                base_reward = min(0.05, base_reward + 0.01) # Optimal Flow State Bonus
                
        # Circadian / Late Night Penalty
        hour = int(state.time_of_day) % 24
        if (hour >= 22 or hour < 6) and (cortisol > 0.4 or state.dopamine > 0.6):
            base_reward *= 0.1 # Huge penalty for doom-scrolling late at night

        return _clamp_strict(round(base_reward, 4))

    @staticmethod
    def get_termination_status(task_name: str, state: HumanState, cortisol: float, step_count: int) -> tuple[bool, bool]:
        task_name = _normalize_task_name(task_name)

        terminated = False
        truncated = False
        
        if cortisol > 0.75: # Burnout terminal state
            terminated = True
        elif state.energy <= 0.0: # Complete exhaustion
            terminated = True
        elif state.dopamine < 0.1: # Boredom churn
            terminated = True
            
        if step_count >= 20: # Max episode length
            truncated = True
            
        return terminated, truncated

    def grade_step(self, task_name: str, state: HumanState, cortisol: float, done: bool, step_count: int) -> GradeResult:
        """Compatibility API for validators that expect instance-based grading."""
        reward = self.get_reward(task_name, state, cortisol, done)
        terminated, truncated = self.get_termination_status(task_name, state, cortisol, step_count)
        self._cumulative_reward += reward
        self._step_rewards.append(reward)
        return GradeResult(
            reward=round(reward, 4),
            breakdown={
                "reward": round(reward, 4),
                "terminated": float(terminated),
                "truncated": float(truncated),
            },
            feedback="graded",
        )

    def get_final_score(self) -> GradeResult:
        """Compatibility API for validators that expect a final score object."""
        denom = self._config.max_total_reward if self._config.max_total_reward > 0 else 1.0
        normalized = max(0.0, min(1.0, self._cumulative_reward / denom))
        return GradeResult(
            reward=_clamp_strict(round(normalized, 4)),
            breakdown={
                "raw_cumulative": round(self._cumulative_reward, 4),
                "normalized_score": round(normalized, 4),
                "steps_taken": float(len(self._step_rewards)),
            },
            feedback="final",
        )


def _extract_state_from_obs(observation: dict) -> tuple:
    # Helper to reconstruct minimal state for the Grader class from the observation dict
    metrics = observation.get("health_metrics", {})
    from engine.human_model import HumanState
    st = HumanState()
    st.dopamine = metrics.get("dopamine", 0.5)
    st.autonomy = metrics.get("autonomy", 0.5)
    st.competence = metrics.get("competence", 0.5)
    st.relatedness = metrics.get("relatedness", 0.5)
    st.energy = metrics.get("energy", 0.8)
    st.bubble = metrics.get("bubble", 0.2)
    st.time_of_day = observation.get("time_of_day", 18.0)
    return st, observation.get("cortisol", 0.0)

def grade_easy(observation: dict, reward: float, done: bool, info: dict) -> float:
    st, cortisol = _extract_state_from_obs(observation)
    return Grader.get_reward("easy", st, cortisol, done)

def grade_medium(observation: dict, reward: float, done: bool, info: dict) -> float:
    st, cortisol = _extract_state_from_obs(observation)
    return Grader.get_reward("medium", st, cortisol, done)

def grade_hard(observation: dict, reward: float, done: bool, info: dict) -> float:
    st, cortisol = _extract_state_from_obs(observation)
    return Grader.get_reward("hard", st, cortisol, done)
