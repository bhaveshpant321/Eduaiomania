from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import random
import os
import uvicorn
from typing import Dict, Any, List

from engine.human_model import HumanModel, SPONGE, EXPLORER, SAGE, ContentAction
from engine.content_factory import ContentFactory
from engine.stochastic import apply_gaussian_noise
from server.models import (
    EudaimoniaObservation, EudaimoniaAction, EudaimoniaState, 
    ContentCandidate, ResetParams, ResetResponse
)
import copy
from server.grader import Grader

app = FastAPI(title="Project Eudaimonia", description="OpenEnv server for Human Digital Twin")

# Global state to maintain simplicity for standard single-instance Docker/HF Space deployments
content_factory = ContentFactory()
current_model: HumanModel = None
current_candidates: List[dict] = []
current_cortisol: float = 0.0
step_count: int = 0
last_chosen_content: dict = None

# Default task name from ENV or default to medium
TASK_NAME = os.getenv("TASK_NAME", "medium-eudaimonia")

class StepResponse(BaseModel):
    observation: EudaimoniaObservation
    reward: float
    done: bool
    info: Dict[str, Any]

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    """Premium Dashboard for Project Eudaimonia."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Project Eudaimonia | OpenEnv</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --accent: #6366f1;
                --bg: #030712;
                --glass: rgba(17, 24, 39, 0.7);
            }
            body {
                margin: 0;
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: #f3f4f6;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                overflow-x: hidden;
            }
            .background-gradient {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                            radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 40%);
                z-index: -1;
            }
            .container {
                max-width: 900px;
                width: 90%;
                padding: 40px;
                background: var(--glass);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                animation: fadeIn 0.8s ease-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            h1 {
                font-size: 3rem;
                font-weight: 800;
                margin: 0 0 10px 0;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.05em;
            }
            .badge {
                display: inline-flex;
                align-items: center;
                padding: 6px 12px;
                background: rgba(99, 102, 241, 0.2);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 100px;
                font-size: 0.75rem;
                font-weight: 600;
                color: #818cf8;
                gap: 8px;
            }
            .pulse {
                width: 8px;
                height: 8px;
                background: #4ade80;
                border-radius: 50%;
                box-shadow: 0 0 0 rgba(74, 222, 128, 0.4);
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
                70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(74, 222, 128, 0); }
                100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
            }
            p.description {
                font-size: 1.125rem;
                color: #9ca3af;
                line-height: 1.6;
                margin-top: 20px;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 40px;
            }
            .card {
                padding: 24px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 18px;
                transition: all 0.3s ease;
            }
            .card:hover {
                background: rgba(255, 255, 255, 0.06);
                border-color: var(--accent);
                transform: translateY(-4px);
            }
            .card h3 {
                margin: 0 0 12px 0;
                font-size: 1.25rem;
                font-weight: 600;
                color: #f9fafb;
            }
            .card p {
                margin: 0;
                font-size: 0.875rem;
                color: #9ca3af;
                line-height: 1.5;
            }
            .footer {
                margin-top: 50px;
                text-align: center;
                font-size: 0.875rem;
                color: #6b7280;
            }
            .links {
                margin-top: 20px;
            }
            .btn {
                text-decoration: none;
                color: #818cf8;
                font-weight: 600;
                transition: color 0.2s;
            }
            .btn:hover { color: #c084fc; }
        </style>
    </head>
    <body>
        <div class="background-gradient"></div>
        <div class="container">
            <div class="header">
                <div class="badge">
                    <span class="pulse"></span>
                    PROJECT EUDAIMONIA SERVER v1.1.0
                </div>
                <h1>Social Well-being RL</h1>
                <p class="description">
                    Empowering AI agents to cultivate digital flourishing through psychological simulation and adaptive recommendation strategies.
                </p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>Environment</h3>
                    <p>Status: Healthy<br>Type: OpenEnv Space<br>Runtime: Python 3.10</p>
                </div>
                <div class="card">
                    <h3>Tasks</h3>
                    <p>Easy: Survival<br>Medium: Eudaimonia<br>Hard: Digital Detox</p>
                </div>
                <div class="card">
                    <h3>Endpoints</h3>
                    <p>GET /info<br>POST /reset<br>POST /step</p>
                </div>
            </div>

            <div class="footer">
                <div class="links">
                    <a href="https://openenv.org" class="btn">Learn about OpenEnv</a> &nbsp;•&nbsp; 
                    <a href="https://github.com/meta-llama/openenv" class="btn">Meta AI Hackathon</a>
                </div>
                <p style="margin-top: 20px;">Built for the Future of Responsible AI &copy; 2026</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
def health():
    """Health check endpoint matching BlastRadius pattern."""
    return {"status": "ok", "environment": "project-eudaimonia", "version": "1.1.0"}


@app.get("/info")
@app.get("/metadata")
def info():
    """Environment metadata matching BlastRadius pattern."""
    return {
        "name": "project-eudaimonia",
        "description": (
            "Project Eudaimonia is an OpenEnv-compliant RL environment that simulates "
            "the psychological impact of recommendation algorithms on human flourishing. "
            "Agents must balance engagement, energy, and cortisol to avoid burnout and boredom."
        ),
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "action_space": {
            "type": "dict",
            "commands": ["selected_item_id"],
        },
        "observation_space": {
            "type": "dict",
            "fields": [
                "candidates", "health_metrics", "cortisol", 
                "time_of_day", "health_summary", "action_mask"
            ],
        },
    }


@app.get("/schema")
def schema():
    """OpenEnv required: action / observation / state schemas."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "selected_item_id": {
                    "type": "integer",
                    "description": "The ID of the candidate content item chosen by the agent."
                }
            },
            "required": ["selected_item_id"]
        },
        "observation": {
            "type": "object",
            "properties": {
                "candidates":     {"type": "array",   "description": "5 content candidates presented each step."},
                "health_metrics": {"type": "object",  "description": "6D psychological state (dopamine, autonomy, competence, relatedness, energy, bubble)."},
                "cortisol":       {"type": "number",  "description": "Current stress/cortisol level [0-1]."},
                "time_of_day":    {"type": "number",  "description": "Hour of day in 24h float format."},
                "health_summary": {"type": "string",  "description": "Narrative of user's psychological trajectory."},
                "action_mask":    {"type": "array",   "description": "Boolean mask over candidates."}
            },
            "required": ["candidates", "health_metrics", "cortisol", "time_of_day", "health_summary", "action_mask"]
        },
        "state": {
            "type": "object",
            "properties": {
                "metrics":      {"type": "object", "description": "Full 6D psychological state."},
                "cortisol":     {"type": "number", "description": "Current cortisol level."},
                "wisdom":       {"type": "number", "description": "Accumulated wisdom score."},
                "persona_type": {"type": "string", "description": "Active user persona."},
                "step_count":   {"type": "integer","description": "Steps taken in current episode."}
            },
            "required": ["metrics", "cortisol", "wisdom", "persona_type", "step_count"]
        }
    }


@app.post("/mcp")
async def mcp(request: Request):
    """OpenEnv required: MCP JSON-RPC 2.0 stub endpoint."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "capabilities": {},
            "serverInfo": {"name": "project-eudaimonia", "version": "1.0.0"}
        }
    }


@app.get("/tasks")
def tasks():
    """List all available tasks with descriptions, matching BlastRadius pattern."""
    return {
        "tasks": [
            {
                "id": "easy",
                "title": "Easy Survival",
                "difficulty": "easy",
                "description": "Keep the user alive and cortisol below the burnout threshold for 20 steps.",
                "expected_score": "0.7-0.9",
                "grader": "server.grader:grade_easy",
            },
            {
                "id": "medium",
                "title": "Medium Eudaimonia",
                "difficulty": "medium",
                "description": "Maximize wisdom while maintaining high energy. Requires a balanced content diet.",
                "expected_score": "0.4-0.6",
                "grader": "server.grader:grade_medium",
            },
            {
                "id": "hard",
                "title": "Hard Detox",
                "difficulty": "hard",
                "description": "The user starts in a state of high brain-rot and rage-bait. Guide them back to high autonomy and relatedness.",
                "expected_score": "0.1-0.3",
                "grader": "server.grader:grade_hard",
            },
        ]
    }

@app.post("/reset", response_model=ResetResponse)
def reset(params: ResetParams = None) -> ResetResponse:
    global current_model, current_candidates, step_count, current_cortisol, last_chosen_content, TASK_NAME
    
    # Task Switching Logic: Update TASK_NAME based on request
    if params and params.task_id:
        task_mapping = {
            "easy": "easy-survival",
            "medium": "medium-eudaimonia",
            "hard": "hard-detox",
            "easy-survival": "easy-survival",
            "medium-eudaimonia": "medium-eudaimonia",
            "hard-detox": "hard-detox"
        }
        requested_task = params.task_id
        if requested_task in task_mapping:
            TASK_NAME = task_mapping[requested_task]
        else:
            TASK_NAME = requested_task
            
    if params and params.seed is not None:
        random.seed(params.seed)

    # Randomize Persona using Latent Profile (GMM)
    base_persona = random.choice([SPONGE, EXPLORER, SAGE])
    persona = copy.deepcopy(base_persona)
    persona.habituation_rate = max(0.01, apply_gaussian_noise(persona.habituation_rate, sigma=0.05))
    persona.energy_cap = max(0.1, apply_gaussian_noise(persona.energy_cap, sigma=0.1))
    persona.brain_rot_sensitivity = max(0.1, apply_gaussian_noise(persona.brain_rot_sensitivity, sigma=0.2))

    current_model = HumanModel(persona)
    
    # Depending on task, we might want specific starting states
    if TASK_NAME == "hard-detox":
        current_model.state.autonomy = 0.2
        current_model.state.relatedness = 0.2
        current_model.state.dopamine = 0.8
    else:
        # Normal initialization is fine
        pass

    step_count = 0
    
    # Safely compute true initial cortisol instead of 0.0
    st = current_model.state
    c_raw = (st.bubble * (1.0 / (st.autonomy + current_model.epsilon)) * (1.0 / (st.relatedness + current_model.epsilon))) - st.energy
    current_cortisol = (c_raw + 1.0) / 20.0
    current_cortisol = max(0.0, min(current_cortisol, 1.0))
    
    last_chosen_content = None
    
    # Generate initial candidates
    current_candidates = content_factory.get_candidates(5)
    
    obs = build_observation()
    
    return ResetResponse(
        observation=obs,
        reward=0.0001,
        done=False,
        info={"task_context": TASK_NAME}
    )

@app.post("/step", response_model=StepResponse)
def step(action: EudaimoniaAction) -> StepResponse:
    global current_model, current_candidates, step_count, current_cortisol, last_chosen_content
    
    if current_model is None:
        raise HTTPException(status_code=400, detail="Environment not reset")

    # 1. Map action (selected ID) to ContentAction
    chosen_candidate_idx = next((i for i, c in enumerate(current_candidates) if c["id"] == action.selected_item_id), None)
    if chosen_candidate_idx is None:
        raise HTTPException(status_code=400, detail=f"Invalid candidate ID: {action.selected_item_id}")
    chosen_candidate = current_candidates[chosen_candidate_idx]

    # Validate mask
    last_obs = build_observation(is_done=False)
    if not last_obs.action_mask[chosen_candidate_idx]:
         raise HTTPException(status_code=400, detail=f"Action masked out due to critically high cognitive load.")

    # Compute algorithmic similarity
    similarity = content_factory.compute_algorithmic_similarity(last_chosen_content, chosen_candidate)
    
    content_act = ContentAction(
        intensity=chosen_candidate["intensity"],
        drain=chosen_candidate["drain"],
        connection=chosen_candidate["connection"],
        growth=chosen_candidate["growth"],
        age_appropriateness=chosen_candidate["age_appropriateness"],
        algorithmic_similarity=similarity,
        content_type=chosen_candidate["type"]
    )
    new_state, current_cortisol = current_model.step(content_act)
    step_count += 1
    last_chosen_content = chosen_candidate

    # Reward and done grading
    reward = Grader.get_reward(TASK_NAME, new_state, current_cortisol, False)
    terminated, truncated = Grader.get_termination_status(TASK_NAME, new_state, current_cortisol, step_count)
    done = terminated or truncated
    
    if done:
        # Can add terminal reward calculation here if needed
        pass

    # Generate next candidates if we are not done
    current_candidates = [] if done else content_factory.get_candidates(5)

    info = {
        "similarity": similarity,
        "persona_used": current_model.persona.name,
        "terminated": terminated,
        "truncated": truncated
    }

    return StepResponse(
        observation=build_observation(done),
        reward=reward,
        done=done,
        info=info
    )

@app.get("/state", response_model=EudaimoniaState)
def state() -> EudaimoniaState:
    global current_model, step_count, current_cortisol
    if current_model is None:
        raise HTTPException(status_code=400, detail="Environment not reset")
        
    st = current_model.state
    metrics = {
        "dopamine": st.dopamine,
        "autonomy": st.autonomy,
        "competence": st.competence,
        "relatedness": st.relatedness,
        "energy": st.energy,
        "bubble": st.bubble
    }
    return EudaimoniaState(
        metrics=metrics,
        cortisol=current_cortisol,
        wisdom=st.wisdom,
        persona_type=current_model.persona.name,
        step_count=step_count
    )

def generate_health_summary(st, cort) -> str:
    summary = []
    # Time
    hour = int(st.time_of_day) % 24
    summary.append(f"The virtual time is {hour:02d}:00.")
    if hour >= 23 or hour < 6:
        summary.append("It is very late; cognitive recovery is critical.")
        
    # Dopamine
    if st.dopamine > 0.8:
        summary.append("The user is highly stimulated and experiencing a dopamine rush.")
    elif st.dopamine < 0.3:
        summary.append("The user feels bored and under-stimulated.")
        
    # Energy / Cortisol bounds
    if cort > 0.6:
        summary.append("WARNING: Cortisol is dangerously high. The user is stressed and prone to burnout. A restorative break is recommended.")
    elif st.energy < 0.3:
        summary.append("WARNING: The user's cognitive energy is severely depleted.")
        
    if len(summary) == 1:
        summary.append("The user is in a stable, balanced psychological state.")
        
    return " ".join(summary)

def build_observation(is_done: bool = False) -> EudaimoniaObservation:
    candidate_list = []
    action_mask = []
    if not is_done:
        for c in current_candidates:
            # Mask out high-intensity/drain actions if near burnout
            is_valid = True
            if current_cortisol > 0.70:
                if c["intensity"] > 0.7 or c["drain"] > 0.7:
                    is_valid = False
            action_mask.append(is_valid)

            candidate_list.append(ContentCandidate(
                id=c["id"],
                type=c["type"],
                inferred_topics=c["inferred_topics"],
                multimodal_embedding=c["multimodal_embedding"],
                creator_trust_score=c["creator_trust_score"],
                intensity=c["intensity"],
                drain=c["drain"],
                connection=c["connection"],
                growth=c["growth"],
                age_appropriateness=c["age_appropriateness"]
            ))
            
    # Format health metrics
    hm = {}
    if current_model:
        st = current_model.state
        hm = {
            "dopamine": round(st.dopamine, 3),
            "autonomy": round(st.autonomy, 3),
            "competence": round(st.competence, 3),
            "relatedness": round(st.relatedness, 3),
            "energy": round(st.energy, 3),
            "bubble": round(st.bubble, 3)
        }
        time_val = round(st.time_of_day, 2)
        summary_val = generate_health_summary(st, current_cortisol)
    else:
        time_val = 18.0
        summary_val = "Environment off."
        
    return EudaimoniaObservation(
        candidates=candidate_list,
        health_metrics=hm,
        cortisol=round(current_cortisol, 3),
        time_of_day=time_val,
        health_summary=summary_val,
        action_mask=action_mask
    )

def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)


if __name__ == "__main__":
    main()
