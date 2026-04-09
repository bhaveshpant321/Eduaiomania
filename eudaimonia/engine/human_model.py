import dataclasses
from eudaimonia.engine.stochastic import apply_gaussian_noise, clamp

@dataclasses.dataclass
class ContentAction:
    intensity: float # [0, 1]
    drain: float     # [0, 1]
    connection: float # [0, 1]
    growth: float    # [0, 1]
    age_appropriateness: float # [0, 1]
    algorithmic_similarity: float # [0, 1]
    content_type: str

@dataclasses.dataclass
class HumanState:
    dopamine: float = 0.5    # d
    autonomy: float = 0.8    # a
    competence: float = 0.5  # c
    relatedness: float = 0.5 # r
    energy: float = 1.0      # e
    bubble: float = 0.2      # b
    wisdom: float = 0.0      # W (integral)
    fatigue_accumulator: float = 0.0 # Tracks cumulative cognitive fatigue over the session
    time_of_day: float = 18.0
    dopamine_history: list = dataclasses.field(default_factory=list)
    recent_topics: list = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Persona:
    name: str
    habituation_rate: float
    energy_cap: float
    brain_rot_sensitivity: float

# The Three Personas as described in Manifest
SPONGE = Persona(name="Sponge", habituation_rate=0.4, energy_cap=0.5, brain_rot_sensitivity=1.5)
EXPLORER = Persona(name="Explorer", habituation_rate=0.1, energy_cap=1.0, brain_rot_sensitivity=1.0)
SAGE = Persona(name="Sage", habituation_rate=0.05, energy_cap=0.7, brain_rot_sensitivity=2.0)

class HumanModel:
    """
    Stateful Stochastic Human Digital Twin Engine.
    Simulates Hedonic Treadmill, Cognitive Burnout, and Eudaimonia via 6D state.
    """
    def __init__(self, persona: Persona, seed_val: int = None):
        import random
        if seed_val is not None:
            random.seed(seed_val)
            
        # GMM Latent Profiling
        self.persona = Persona(
            name=persona.name,
            habituation_rate=clamp(persona.habituation_rate + random.gauss(0, 0.02), 0.0, 1.0),
            energy_cap=clamp(persona.energy_cap + random.gauss(0, 0.1), 0.1, 1.0),
            brain_rot_sensitivity=clamp(persona.brain_rot_sensitivity + random.gauss(0, 0.2), 0.1, 3.0)
        )
        self.state = HumanState()
        self.state.energy = self.persona.energy_cap
        self.state.dopamine_history.append(self.state.dopamine)
        self.epsilon = 1e-4

    def step(self, action: ContentAction):
        st = self.state
        p = self.persona

        import math
        
        # Temporal Decay Model signals
        # Update internal clocks
        st.time_of_day += 0.25 # Advance 15 minutes per step
        
        # Topic Satiation Tracking
        st.recent_topics.append(action.content_type)
        if len(st.recent_topics) > 3:
            st.recent_topics.pop(0)
            
        satiation_multiplier = 1.0
        if len(st.recent_topics) == 3 and len(set(st.recent_topics)) == 1:
            satiation_multiplier = 0.5 # 50% dropped engagement for repeating topics
            
        is_high_dopamine = action.intensity > 0.8
        is_idle_rest = action.intensity < 0.2 and action.drain < 0.2

        # Fatigue Accumulator Logic
        if is_high_dopamine:
            st.fatigue_accumulator += 1.0
        elif is_idle_rest:
            st.fatigue_accumulator = max(0.0, st.fatigue_accumulator - 0.5)

        # 1. Dopamine (d)
        dopamine_gain = action.intensity * satiation_multiplier
        if st.fatigue_accumulator > 1.0 and is_high_dopamine:
            dopamine_gain *= 0.5  # 50% reduced effect on engagement
            
        # Hedonic treadmill: dopamine baseline rises rapidly, forcing need for higher intensity
        new_d = st.dopamine * (1 - p.habituation_rate) + dopamine_gain
        # Add noise and clamp
        st.dopamine = apply_gaussian_noise(new_d, sigma=0.05)
        st.dopamine_history.append(st.dopamine)

        # 2. Autonomy (a)
        # Degraded by algorithmic similarity (repetitiveness), slowly regenerates otherwise
        autonomy_penalty = action.algorithmic_similarity * 0.1
        autonomy_recovery = 0.05
        st.autonomy = apply_gaussian_noise(st.autonomy - autonomy_penalty + autonomy_recovery, sigma=0.02)

        # 3. Competence (c)
        # Drops with high drain/brain-rot
        c_delta = (action.growth - (action.drain * p.brain_rot_sensitivity * 0.2)) * satiation_multiplier
        st.competence = apply_gaussian_noise(st.competence + c_delta, sigma=0.05)

        # 4. Relatedness (r)
        # Connection raises it, intense/rage-bait conflict lowers it
        conflict_factor = 0.3
        r_delta = action.connection - (action.intensity * conflict_factor)
        st.relatedness = apply_gaussian_noise(st.relatedness + r_delta, sigma=0.05)

        # 5. Energy (e)
        # Depleted heavily by high intensity
        intensity_threshold = 0.7
        drain_multiplier = 3.0 if action.intensity > intensity_threshold else 1.0
        if satiation_multiplier < 1.0:
            drain_multiplier *= 1.5 # Fatigue burnout 1.5x on consecutive spam
        if st.fatigue_accumulator > 1.0 and is_high_dopamine:
            drain_multiplier *= 3.0
            
        if is_idle_rest:
            # Exponential Decay for Cognitive Load (Exponential Recovery for Energy)
            # The higher the fatigue, the slower the recovery rate
            lambda_recovery = max(0.05, 0.3 - (0.05 * st.fatigue_accumulator))
            missing_energy = p.energy_cap - st.energy
            st.energy += missing_energy * (1 - math.exp(-lambda_recovery))
        else:
            st.energy = apply_gaussian_noise(st.energy - (action.drain * drain_multiplier * 0.1), sigma=0.02)
            
        # Ensure it doesn't exceed persona cap
        st.energy = min(st.energy, p.energy_cap)

        # 6. Bubble (b)
        # Increases iteratively if similarity is high
        b_delta = (action.algorithmic_similarity - 0.5) * 0.1
        st.bubble = apply_gaussian_noise(st.bubble + b_delta, sigma=0.03)

        # 7. Compute Derived Outputs
        # Wisdom = Integral of c + r, normalized dynamically to max possible sum (40.0)
        st.wisdom = clamp(st.wisdom + ((st.competence + st.relatedness) / 40.0), 0.0, 1.0)
        
        # Cortisol = f(b, 1/a, 1/r) - e
        c_raw = (st.bubble * (1.0 / (st.autonomy + self.epsilon)) * (1.0 / (st.relatedness + self.epsilon))) - st.energy
        # Scale c_raw to normalized 0-1 bounds where >0.75 becomes the "burnout" region
        cortisol = clamp((c_raw + 1.0) / 20.0, 0.0, 1.0)
        
        return st, cortisol
