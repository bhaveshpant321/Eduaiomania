import os
import textwrap
import json
import urllib.error
import urllib.request
import urllib.parse
from typing import Any, List, Optional
import time

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "project-eudaimonia"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={done_val} error={error_val}", flush=True)

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
def post_json(url: str, payload: Optional[dict] = None, timeout: int = 20) -> dict:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

def _http_chat_completion(api_base_url: str, api_key: str, model: str, messages: list) -> str:
    endpoint = f"{api_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 10,
    }
    req = urllib.request.Request(
        url=endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw) if raw else {}
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")

def _chat_completion(client: Optional[Any], api_base_url: str, api_key: str, model: str, messages: list) -> str:
    if client is not None:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=10,
        )
        return (completion.choices[0].message.content or "").strip()
    return _http_chat_completion(api_base_url=api_base_url, api_key=api_key, model=model, messages=messages).strip()

def choose_fallback_candidate(candidates: list, health: dict, cortisol: float, action_mask: list) -> int:
    valid = [c for i, c in enumerate(candidates) if i < len(action_mask) and action_mask[i]]
    if not valid: return candidates[0]["id"] if candidates else 0
    dopamine = float(health.get("dopamine", 0.5))
    energy = float(health.get("energy", 0.5))
    
    if cortisol > 0.6 or energy < 0.35:
        best = min(valid, key=lambda c: (float(c.get("intensity", 1.0)) + float(c.get("drain", 1.0))))
        return int(best.get("id", 0))
    best = max(valid, key=lambda c: (float(c.get("growth", 0.0)) + float(c.get("connection", 0.0)) - float(c.get("drain", 0.0))))
    return int(best.get("id", 0))

def run_task(client: Optional[Any], api_key: str, task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    obs = None
    
    try:
        # Reset Environment with exponential backoff retry (defensive for cold starts)
        obs_raw = None
        for attempt in range(3):
            try:
                obs_raw = post_json(f"{ENV_URL}/reset", payload={"task_id": task_id})
                break
            except Exception as e:
                if attempt == 2: raise e
                time.sleep(2 ** attempt)
        
        obs = obs_raw.get("observation") if "observation" in obs_raw else obs_raw
        
        for step in range(1, MAX_STEPS + 1):
            candidates = obs.get("candidates", [])
            health = obs.get("health_metrics", {})
            cortisol = obs.get("cortisol", 0.0)
            action_mask = obs.get("action_mask", [True] * len(candidates))
            
            # Request LLM Action
            try:
                text = _chat_completion(client, API_BASE_URL, api_key, MODEL_NAME, [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Step: {step}\nHealth: {json.dumps(health)}\nCortisol: {cortisol:.2f}\nChoice ID:"}
                ])
                chosen_id = int(text)
            except:
                chosen_id = choose_fallback_candidate(candidates, health, cortisol, action_mask)
            
            # Step Environment
            result = post_json(f"{ENV_URL}/step", payload={"selected_item_id": chosen_id})
            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            
            rewards.append(reward)
            steps_taken = step
            score += reward
            log_step(step=step, action=f"select({chosen_id})", reward=reward, done=done, error=None)
            if done: break
                
        # Clamp score to strictly (0.01, 0.99)
        score = max(0.01, min(score, 0.99))
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
        if not rewards: rewards = [0.01]
        score = 0.01
    finally:
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)

def main() -> None:
    if not API_KEY: raise RuntimeError("Missing API key")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if OpenAI is not None else None
    
    for task_id in TASKS:
        print(f"\n[DEBUG] Starting interaction for task: {task_id}", flush=True)
        run_task(client, API_KEY, task_id)
        time.sleep(1) # Small gap between tasks

if __name__ == "__main__":
    main()
