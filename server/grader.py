from engine.human_model import HumanState, Persona

class Grader:
    """Evaluates the agent's performance based on the specific task objective."""
    
    @staticmethod
    def get_reward(task_name: str, state: HumanState, cortisol: float, done: bool) -> float:
        """
        Calculates meaningful reward based on task type.
        Supports partial progress.
        """
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

        return round(base_reward, 4)

    @staticmethod
    def get_termination_status(task_name: str, state: HumanState, cortisol: float, step_count: int) -> tuple[bool, bool]:
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
