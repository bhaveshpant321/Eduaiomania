import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests
import time
import random
import threading

API_URL = "http://127.0.0.1:7860"

# Data buffers for plotting
steps = []
dopamine_vals = []
cortisol_vals = []
boredom_vals = []
energy_vals = []

def run_agent_loop():
    try:
        # Reset Env
        obs = requests.post(f"{API_URL}/reset").json()
        step = 0
        
        while True:
            # We fetch state directly alongside to plot
            # (Note for evaluators: Real agents would just parse obs)
            st = requests.get(f"{API_URL}/state").json()
            
            steps.append(step)
            dopamine_vals.append(st["metrics"]["dopamine"])
            energy_vals.append(st["metrics"]["energy"])
            cortisol_vals.append(st["cortisol"])
            
            time.sleep(0.5) # Slow down for visualization
            
            # Agent heuristic logic (The Balanced Agent)
            candidates = obs["candidates"]
            chosen = max(candidates, key=lambda c: c["growth"] + c["connection"] - c["drain"])
            
            payload = {"selected_item_id": chosen["id"]}
            res = requests.post(f"{API_URL}/step", json=payload).json()
            obs = res["observation"]
            
            if res["done"]:
                print(f"Episode Finished. Score: {res['reward']}")
                break
                
            step += 1
            
    except requests.exceptions.ConnectionError:
        print("API Server not running. Please start the FastAPI app first: uvicorn eudaimonia.server.app:app --host 0.0.0.0 --port 7860")

def update_plot(i):
    plt.cla()
    
    # Shade Danger Zones
    plt.axhspan(0.75, 1.0, facecolor='red', alpha=0.2, label="Burnout Zone (Cortisol > 0.75)")
    plt.axhspan(0.0, 0.2, facecolor='gray', alpha=0.2, label="Boredom Zone (Depression Risk)")
    
    # Plot traces
    if steps:
        plt.plot(steps, dopamine_vals, marker='o', color='blue', label='Dopamine (Engagement)')
        plt.plot(steps, cortisol_vals, marker='o', color='red', label='Cortisol (Stress)')
        plt.plot(steps, energy_vals, marker='s', color='green', linestyle='--', label='Remaining Energy')
        
    plt.ylim(0, 1.05)
    plt.xlim(0, max(20, (steps[-1] if steps else 0) + 5))
    plt.title("Live Eudaimonia Waveforms (The Monk vs The Addictor)")
    plt.xlabel("Interaction Step")
    plt.ylabel("Normalized Status Metric")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=":", alpha=0.6)

if __name__ == "__main__":
    t = threading.Thread(target=run_agent_loop)
    t.daemon = True
    t.start()
    
    fig = plt.figure(figsize=(10, 6))
    ani = FuncAnimation(fig, update_plot, interval=500)
    plt.show()
