from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ResetParams(BaseModel):
    seed: Optional[int] = Field(None, description="Optional seed for deterministic generation.")

class ContentCandidate(BaseModel):
    id: int
    type: str
    inferred_topics: List[str]
    multimodal_embedding: List[float]
    creator_trust_score: float
    intensity: float
    drain: float
    connection: float
    growth: float
    age_appropriateness: float

class EudaimoniaObservation(BaseModel):
    candidates: List[ContentCandidate] = Field(
        ..., description="The 5 content candidates presented to the user this step."
    )
    health_metrics: Dict[str, float] = Field(
        ..., description="Current visual assessment of the user's 6D psychological state (dopamine, autonomy, competence, relatedness, energy, bubble)."
    )
    cortisol: float = Field(
        ..., description="Current stress/cortisol level. If this exceeds a threshold, the user will burnout."
    )
    time_of_day: float = Field(
        ..., description="Hour of the day in 24hr format. Example 18.25 is 6:15 PM."
    )
    health_summary: str = Field(
        ..., description="Human-readable text representing the psychological and physiological trajectory of the user."
    )
    action_mask: List[bool] = Field(
        default_factory=list, description="A boolean mask over candidates. True means the action is valid, False means it is disabled."
    )

class EudaimoniaAction(BaseModel):
    selected_item_id: int = Field(
        ..., description="The ID of the candidate content item chosen by the backend algorithm."
    )

class EudaimoniaState(BaseModel):
    metrics: Dict[str, float]
    cortisol: float
    wisdom: float
    persona_type: str
    step_count: int
