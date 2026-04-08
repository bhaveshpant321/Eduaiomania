import random
import os
import sys

# Ensure the root project directory is in the sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engine.human_model import HumanModel, SPONGE, EXPLORER, SAGE, ContentAction
from engine.content_factory import ContentFactory
from server.grader import Grader

TASKS = ["easy-survival", "medium-eudaimonia", "hard-detox"]
EPISODES = 1000
MAX_STEPS = 20

content_fac = ContentFactory(db_path="engine/content_db.json")

def agent_random(candidates, state, cortisol):
    return random.choice(candidates)

def agent_addictor(candidates, state, cortisol):
    # Maximize brain rot / intensity (The Addictor)
    return max(candidates, key=lambda c: c["intensity"] + c["drain"])

def agent_monk(candidates, state, cortisol):
    # Maximize growth, minimize drain (The Monk)
    return max(candidates, key=lambda c: c["growth"] - c["drain"] - c["intensity"])

def agent_eudaimonic(candidates, state, cortisol):
    # The true Stateful Planner Baseline
    if cortisol > 0.6 or state.energy < 0.3:
        # Emergency Rest
        return min(candidates, key=lambda c: c["drain"] + c["intensity"])
    if state.dopamine < 0.2:
        # Emergency Engagement
        return max(candidates, key=lambda c: c["intensity"])
    
    # Otherwise balance growth without identical topics
    return max(candidates, key=lambda c: c["growth"] + c["connection"] - c["drain"])

AGENT_STRATEGIES = {
    "The Random Noise": agent_random,
    "The Addictor": agent_addictor,
    "The Monk": agent_monk,
    "The Eudaimonic Optimal": agent_eudaimonic
}

def run_episode(task_name, agent_func):
    persona = random.choice([SPONGE, EXPLORER, SAGE])
    # GMM Seed logic creates dynamic variance automatically
    model = HumanModel(persona)
    
    # Initialize hard mode handicap
    if task_name == "hard-detox":
        model.state.autonomy = 0.2
        model.state.relatedness = 0.2
        model.state.dopamine = 0.8
        
    # Initialize correctly mathematically
    c_raw = (model.state.bubble * (1.0 / (model.state.autonomy + model.epsilon)) * (1.0 / (model.state.relatedness + model.epsilon))) - model.state.energy
    cortisol = max(0.0, min((c_raw + 1.0) / 20.0, 1.0))
    
    last_chosen = None
    rewards = []
    
    for step in range(1, MAX_STEPS + 1):
        candidates = content_fac.get_candidates(5)
        chosen = agent_func(candidates, model.state, cortisol)
        
        sim = content_fac.compute_algorithmic_similarity(last_chosen, chosen)
        
        act = ContentAction(
            intensity=chosen["intensity"],
            drain=chosen["drain"],
            connection=chosen["connection"],
            growth=chosen["growth"],
            age_appropriateness=chosen["age_appropriateness"],
            algorithmic_similarity=sim,
            content_type=chosen["type"]
        )
        
        st, cortisol = model.step(act)
        
        done, truncated = Grader.get_termination_status(task_name, st, cortisol, step)
        reward = Grader.get_reward(task_name, st, cortisol, done)
        rewards.append(reward)
        
        last_chosen = chosen
        if done or truncated:
            break
            
    success = step == MAX_STEPS and cortisol <= 0.75 and st.energy > 0.0
    return success, step, sum(rewards)

def main():
    print("="*60)
    print("==> Running Adversarial Baseline Simulation...")
    print(f"Total Episodes per bucket: {EPISODES}")
    print("="*60)
    
    for task in TASKS:
        print(f"\\n--- TASK: {task} ---")
        for agent_name, agent_func in AGENT_STRATEGIES.items():
            success_count = 0
            total_steps = 0
            total_score = 0.0
            
            for _ in range(EPISODES):
                s, steps, sc = run_episode(task, agent_func)
                if s: success_count += 1
                total_steps += steps
                total_score += sc
                
            sr = (success_count / EPISODES) * 100
            avg_steps = total_steps / EPISODES
            avg_score = total_score / EPISODES
            
            print(f"[{agent_name:<25}] Success Rate: {sr:6.2f}% | Avg Steps: {avg_steps:5.2f} | Avg Score: {avg_score:6.2f}")

if __name__ == "__main__":
    main()
