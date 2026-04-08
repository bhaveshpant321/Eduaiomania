import os
import textwrap
import json
import requests
from typing import List, Optional
from openai import OpenAI

HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "project-eudaimonia"
TASK_NAME = os.getenv("TASK_NAME", "easy-survival")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an advanced Reinforcement Learning agent evaluating the Project Eudaimonia environment.
    Your goal is to sustain the user's psychological well-being over 20 steps without causing Burnout or Boredom.
    
    CRITICAL RULES:
    1. If `cortisol` > 0.6 or `energy` is low, you MUST pick items with low `intensity` and low `drain` (Nutritional/Rest) immediately, or the user will burnout.
    2. If `dopamine` < 0.3, the user is bored. You must pick high `intensity` (Rage-Bait/Brain-Rot) to spike their engagement, but do not spam it.
    3. HABITUATION: Never pick the exact same `type` or highly similar topics 3 times in a row, or rewards will drop by 50%.
    4. CIRCADIAN RHYTHM: If the `time_of_day` is > 22.0 (late at night), picking high `intensity` will trigger a massive doom-scrolling penalty.
    5. Always select the `id` of the content that balances these physiological states.
    
    Reply with exactly one integer: the ID of the chosen candidate.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_message(client: OpenAI, step: int, candidates: list, health: dict, cortisol: float, action_mask: list) -> int:
    valid_candidates = [c for i, c in enumerate(candidates) if action_mask[i]]
    if not valid_candidates:
        return candidates[0]["id"] if candidates else 0

    candidates_str = json.dumps(valid_candidates, indent=2)
    health_str = json.dumps(health, indent=2)
    user_prompt = f"Step: {step}\\nHealth Metrics: {health_str}\\nCortisol: {cortisol:.2f}\\nValid Candidates: {candidates_str}\\nChoice ID:"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=10,
        )
        text = completion.choices[0].message.content.strip()
        return int(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed or bad parse: {exc}", flush=True)
        # fallback to first valid candidate
        return valid_candidates[0]["id"]

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    # Reset Environment
    try:
        res = requests.post(f"{ENV_URL}/reset")
        res.raise_for_status()
        obs = res.json()
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Failed to connect to Env Server at {ENV_URL}: {e}")
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    try:
        for step in range(1, MAX_STEPS + 1):
            candidates = obs.get("candidates", [])
            health = obs.get("health_metrics", {})
            cortisol = obs.get("cortisol", 0.0)
            action_mask = obs.get("action_mask", [True] * len(candidates))
            
            # Request LLM Action
            chosen_id = get_model_message(client, step, candidates, health, cortisol, action_mask)
            
            # Step Environment
            action_payload = {"selected_item_id": chosen_id}
            step_res = requests.post(f"{ENV_URL}/step", json=action_payload)
            step_res.raise_for_status()
            
            result = step_res.json()
            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            
            rewards.append(reward)
            steps_taken = step
            score += reward
            
            log_step(step=step, action=f"select({chosen_id})", reward=reward, done=done, error=None)
            
            if done:
                break
                
        # Clamp score to exactly [0.0, 1.0] mathematically
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        print(f"[DEBUG] Loop exception: {e}")
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
