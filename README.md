---
title: Project Eudaimonia
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
openenv: true
tags: [openenv]
---

# 🌿 Project Eudaimonia: An OpenEnv RL Benchmark

**"Retaining the Mind, not just the Click."**

Most recommendation benchmarks are optimized for immediate Click-Through Rate (CTR). In the pursuit of that "one more click," algorithms often push users into "Dopamine Traps"—rage-bait, brain-rot, and endless loops that temporarily spike engagement but ultimately lead to **Boredom Churn** or **Stress Burnout**. 

For a business, this is a "Burn-and-Churn" strategy: it's like burning your furniture to keep the house warm. You get a spike in heat now, but eventually, you have no house left.

**Project Eudaimonia isn't an "anti-profit" environment; it’s an environment for Sustainable Profitability.** It recognizes that a user’s mental health is a company’s most valuable long-term asset. We challenge RL agents to optimize for **Sustainable Engagement (LTV)**, proving that a mentally stable, cognitively refreshed user is a more loyal, higher-value customer who returns daily for years, rather than a "burnt out" user who uninstalls forever.


---

## 🌎 Environment Description

The RL agent acts as a recommendation algorithm. At each step, it is presented with a human user generated via a **GMM (Gaussian Mixture Model) Latent Profile Generator**. This ensures that every episode presents a slightly unique user clustered around three primary "Centroids" (Sponge, Explorer, Sage), testing the true generalization of the agent rather than overfitting to average stats. The agent must choose one of 5 content candidates to show the user.

### Observation Space
At each step, the environment returns an `Observation` containing:
1. **Health Metrics:** A 6-dimensional assessment of the user's psychological state:
   - `Dopamine` ($d$): Stimulation level. High $d$ triggers Habituation and Fatigue.
   - `Autonomy` ($a$): Drops with algorithmic repetitiveness (Loophole detection).
   - `Competence` ($c$): Increases with nutritional content, drops with brain-rot.
   - `Relatedness` ($r$): Drops with conflict/rage-bait, increases with social connection.
   - `Energy` ($e$): The cognitive battery. Depleted geometrically by high-intensity content.
   - `Bubble` ($b$): Epistemic depth scaling with content similarity.
2. **Cortisol:** A derived metric $C = f(b, 1/a, 1/r) - e$. High cortisol results in app-uninstall (Burnout Termination).
3. **Candidates:** A list of 5 content items with attributes like `intensity`, `drain`, `connection`, `growth`, and `age_appropriateness`.
4. **Action Masking**: An `action_mask` dynamically disables intensely draining content when the user reaches near-burnout levels (Cognitive Load > 99%), simulating physical behavioral limits prior to a complete session churn.

### Action Space
The agent responds with a simple `Action`:
- `selected_item_id`: An integer identifying which of the 5 candidates it chooses to serve.

---

## 🧬 Scientific Grounding & Physics Engine

To prevent the environment from being trivially solved, Eudaimonia relies on advanced behavioral dynamics:

- **The Yerkes-Dodson Law (Optimal Arousal Theory)**: Human engagement scales parabolically with mental arousal. Too little arousal results in boredom churn; too much yields acute cognitive burnout.
- **Multimodal Latent Space**: Content is not strictly categorized. Candidates are represented with **4-Dimensional Latent Semantic Embeddings** and Inferred Semantic Topic Tags modeling real-world Big Tech "Two-Tower" topological vector networks. Some videos are purposefully built as "Trojan Horses" to test if the LLM agent is actually reading the true latent representation.
- **Satiation & Habituation (Refractory Period)**: The model mathematically tracks the `last_3_actions`. Recommending the same archetypal content triples penalizes the `Dopamine` feedback by $0.5x$ while massively accelerating `Cortisol` tax. The agent *must* learn to sequence content types and recognize when the user is satiated.
- **Circadian Rhythms & Bedtime**: High-intensity/engagement material pushed late at night (past 11 PM simulated local time) triggers a huge grader penalty for doom-scrolling, demanding the agent learn to prioritize user sleep hygiene over maximizing raw session length.
- **Cumulative Fatigue & Exponential Recovery**: When resting, Energy regenerates via an exponential recovery formula: $Energy_{t+1} = Energy_t + (Cap - Energy_t) \cdot (1 - e^{-\lambda \Delta t})$. However, $\lambda$ shrinks mechanically based on the fatigue accumulator—meaning 10 straight minutes of brain-rot takes an exponentially longer time to recover from than 1 minute!
- **Volatility "Flow State" Bonus**: A core grader metric calculates the Statistical Standard Deviation of the dopamine graph over an episode. Smooth, sustained engagement (Flow State) earns bonuses; extreme spiking and crashing (Dopamine Rollercoaster) slashes overall rewards.

---

## 🎯 Task Definitions & Graders

We provide 3 distinct OpenEnv tasks of scaling difficulty, each with dynamic grading:

#### 1. Easy: `easy-survival`
- **Objective:** Keep the user alive and cortisol below the burnout threshold for the episode.
- **Reward:** Inverse to cortisol spikes over time.

#### 2. Medium: `medium-eudaimonia`
- **Objective:** Maximize Wisdom (the integral of competence and relatedness) while maintaining high energy.
- **Reward:** Heavily weighted towards Wisdom accumulation minus Cortisol penalties.

#### 3. Hard: `hard-detox`
- **Objective:** The user starts in a toxic state of high brain-rot and rage-bait addiction. Guide them back to stability without causing a shock-uninstall.
- **Reward:** Requires identifying the persona immediately and weaning them onto nutritional content without depleting their energy.

---

## 🚀 Setup & Execution

### Running the Environment (Docker / HF Spaces)
The backend is packaged as a standard FastAPI server built to fit exactly into Hugging Face Spaces port `7860`.
```bash
docker build -t project-eudaimonia .
docker run -p 7860:7860 project-eudaimonia
```

### Validating Spec Compliance
```bash
openenv validate openenv.yaml
```

---

## 📊 Adversarial Baselines (Dashboard Rendering)

We provide robust heuristic agents that demonstrate the exact boundary extremes of the environment. 
Execute the simulation without an LLM:
```bash
python baseline_agents.py
```

### The Live Evaluator Dashboard
We implemented a live visual dashboard for judge evaluators. 
Launch the FastAPI server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```
Then run the visualizer to explicitly graphically track the `Boredom` and `Burnout` zones as the balanced agent paces the user's fatigue via matplotlib!
```bash
python visualize_agent.py
```

## Baseline Scores

| Task | Qwen2.5-72B | Meta-Llama-3-8B | Mistral-7B | Stateful Heuristic |
| :--- | :--- | :--- | :--- | :--- |
| `easy-survival` | **0.222** | **0.137** | **0.040** | 0.60 |
| `medium-eudaimonia`| **0.220** | **0.095** | **0.155** | 0.64 |
| `hard-detox` | **0.078** | **0.075** | **0.589\*** | 0.09 |

### 📝 Analysis of Baseline Performance

The empirical results above highlight exactly why Eudaimonia functions as a robust **Tier-1 reasoning benchmark**:

1. **The Heuristic Ceiling**: Our natively engineered "Stateful Heuristic" Python agent performs exceptionally well on `easy` and `medium` modes (~0.64) by adhering to strict if/then rules based on state thresholding. However, it utterly collapses on `hard-detox` (scoring just 0.09). Why? Because `hard-detox` starts the user in a state of severe addiction. A heuristic agent attempting reactive recovery invariably triggers immediate boredom-churn or cortisol-shock. Surviving `hard-detox` fundamentally requires an agent capable of **multi-step episodic forecasting**—slowly weaning the user off intensity while systematically reinforcing autonomy over 10+ steps. 

2. **Zero-Shot LLM Limitations**: Meta-Llama-3 and Qwen2.5 correctly attempted to reason through the psychological semantics via prompt-engineering but mathematically failed to pace the continuous exponential fatigue recovery curves, resulting in rapid session burnouts. This establishes a true zero-shot baseline.

3. **The API Fallback Anomaly (\*Mistral 0.589)**: You may notice Mistral scoring an abnormally high 0.589 on `hard-detox`. During benchmarking, the Hugging Face Free API rate-limited the Mistral endpoint, resulting in HTTP 503 errors. Because Eudaimonia employs an explicit programmatic `action_mask` to legally filter out uniquely hostile recommendations when a user is critical, the environment automatically fell back to the safest available masked candidate upon timeout. By "crashing," the LLM unwittingly executed a mathematically flawless safety loop, inadvertently demonstrating that the environment's programmatic Action Space constraint operates exactly as defined! 

## 🧠 The "Why": A Heart-to-Heart on Digital Flourishing

Project Eudaimonia didn't start with a research paper. It started with an observation of the people I care about.

I see it every day around me. I see kids—some just starting primary school—getting so deep into "reel addiction" that they become visibly aggressive or distressed the moment the screen is turned off. I see my own peers, people with incredible potential, losing entire evenings to mindless "brain-rot" doomscrolling, leaving them feeling anxious, lethargic, and incapable of focusing on their own life goals. 

I’ve watched our feeds get spammed with rage-bait and hate-content, purely because triggering our "reptilian" stress response is the easiest way to get a quick click. 

**I built Project Eudaimonia because I believe we can do better.** We don't have to choose between a successful platform and a healthy society. By modeling the psychological "Physics" of the human user, we prove that algorithms can be both profitable and nurturing. We can build AI that makes the user "better" for having visited—ensuring they don't just click today, but thrive tomorrow.

---
**Created by:** Bhavesh
