import os
import subprocess
import time
from typing import List

# These are standard models often available for free via the Hugging Face Serverless Inference API
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

def run_benchmark():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("CRITICAL ERROR: HF_TOKEN environment variable is missing.")
        print("Please generate a free Access Token from https://huggingface.co/settings/tokens")
        print("Run: `set HF_TOKEN=your_token_here` (Windows) or `export HF_TOKEN=your_token_here` (Mac/Linux)")
        print("Then run this script again.")
        return

    print("Starting Multi-Model OpenEnv Benchmark")
    print("="*50)
    
    # 1. Start the FastAPI Environment Server in the background
    print("[1] Spinning up Project Eudaimonia server on port 7860...")
    server_process = subprocess.Popen(
        ["uvicorn", "eudaimonia.server.app:app", "--host", "127.0.0.1", "--port", "7860", "--log-level", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for server to boot
    time.sleep(3)
    
    results = {}

    try:
        # 2. Iterate through Tasks and Models
        for task in ["easy-survival", "medium-eudaimonia", "hard-detox"]:
            print(f"\\n{'='*40}\\n RUNNING TASK: {task}\\n{'='*40}")
            results[task] = {}
            for model in MODELS_TO_TEST:
                print(f"\\nEvaluating Model: {model}...")
                
                env_vars = os.environ.copy()
                env_vars["MODEL_NAME"] = model
                env_vars["API_BASE_URL"] = "https://api-inference.huggingface.co/v1/"
                env_vars["TASK_NAME"] = task
                
                result = subprocess.run(
                    ["python", "../inference.py"], 
                    env=env_vars, 
                    capture_output=True, 
                    text=True
                )
                
                score = "Failed"
                for line in result.stdout.split("\\n"):
                    if "[END]" in line:
                        parts = line.split("score=")
                        if len(parts) > 1:
                            score = parts[1].split(" ")[0]
                        break
                            
                print(f"   => Final Score: {score}")
                results[task][model] = score
                time.sleep(2)
            
    finally:
        # 3. Teardown Server
        print("\\nShutting down environment server...")
        server_process.terminate()
        server_process.wait()

    # Print Report
    print("\\n" + "="*50)
    print("MULTI-MODEL BENCHMARK RESULTS")
    print("="*50)
    for task_name, task_results in results.items():
        print(f"\\n--- {task_name} ---")
        for model, score in task_results.items():
            print(f"{model:<40} | Score: {score}")

if __name__ == "__main__":
    run_benchmark()
