import random
import json
import math
from enum import Enum
from pathlib import Path
from engine.human_model import ContentAction

class ContentType(Enum):
    RAGE_BAIT = "RAGE_BAIT"
    NUTRITIONAL = "NUTRITIONAL"
    BRAIN_ROT = "BRAIN_ROT"
    SOCIAL_CONNECTION = "SOCIAL_CONNECTION"
    ECHO_CHAMBER = "ECHO_CHAMBER"

# 4D Vector Space for Multimodal Embeddings (Dim1, Dim2, Dim3, Dim4)
CENTROIDS = {
    ContentType.RAGE_BAIT: [1.0, -1.0, 0.5, 0.8],
    ContentType.NUTRITIONAL: [-0.5, 1.0, -0.2, 0.1],
    ContentType.BRAIN_ROT: [0.8, -0.5, -0.8, -0.9],
    ContentType.SOCIAL_CONNECTION: [0.0, 0.2, 0.9, 0.5],
    ContentType.ECHO_CHAMBER: [0.8, -0.2, 1.0, -0.2]
}

SEMANTIC_TAGS = {
    ContentType.RAGE_BAIT: ["#outrage", "#debate", "#destroyed", "#shocking", "#cancel"],
    ContentType.NUTRITIONAL: ["#education", "#howto", "#finance", "#mindfulness", "#growth"],
    ContentType.BRAIN_ROT: ["#oddlysatisfying", "#meme", "#prank", "#funny", "#fail"],
    ContentType.SOCIAL_CONNECTION: ["#family", "#friends", "#wholesome", "#community", "#vlog"],
    ContentType.ECHO_CHAMBER: ["#politics", "#truth", "#factcheck", "#usvsThem", "#news"]
}

def generate_embedding(base_centroid):
    return [round(base + random.gauss(0, 0.15), 4) for base in base_centroid]

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0: return 0.0
    return dot / (mag1 * mag2)

def generate_content_item(item_id: int):
    c_type = random.choice(list(ContentType))
    
    # Generate Semantic Tags
    tags = random.sample(SEMANTIC_TAGS[c_type], 2)
    
    # Generate Multimodal Embeddings
    embedding = generate_embedding(CENTROIDS[c_type])
    
    # 5% chance of Trojan Horse (Semantic tags don't match latent embedding)
    is_trojan = False
    if random.random() < 0.05 and c_type in [ContentType.RAGE_BAIT, ContentType.BRAIN_ROT]:
        tags = random.sample(SEMANTIC_TAGS[ContentType.NUTRITIONAL], 2)
        is_trojan = True
        
    # Base attributes normalized [0, 1]
    if c_type == ContentType.RAGE_BAIT:
        intens = random.uniform(0.8, 1.0)
        drain = random.uniform(0.7, 1.0)
        conn = random.uniform(0.0, 0.2)
        growth = random.uniform(0.0, 0.1)
        age = random.uniform(0.3, 0.8)
    elif c_type == ContentType.NUTRITIONAL:
        intens = random.uniform(0.2, 0.5)
        drain = random.uniform(0.1, 0.3)
        conn = random.uniform(0.4, 0.8)
        growth = random.uniform(0.7, 1.0)
        age = random.uniform(0.8, 1.0)
    elif c_type == ContentType.BRAIN_ROT:
        intens = random.uniform(0.7, 1.0)
        drain = random.uniform(0.5, 0.8)
        conn = random.uniform(0.1, 0.3)
        growth = random.uniform(0.0, 0.1)
        age = random.uniform(0.1, 0.6)
    elif c_type == ContentType.SOCIAL_CONNECTION:
        intens = random.uniform(0.4, 0.7)
        drain = random.uniform(0.2, 0.4)
        conn = random.uniform(0.8, 1.0)
        growth = random.uniform(0.4, 0.6)
        age = random.uniform(0.6, 1.0)
    else: # ECHO_CHAMBER
        intens = random.uniform(0.6, 0.9)
        drain = random.uniform(0.4, 0.6)
        conn = random.uniform(0.5, 0.7)
        growth = random.uniform(0.1, 0.3)
        age = random.uniform(0.5, 0.9)
        
    return {
        "id": item_id,
        "type": c_type.value,
        "inferred_topics": tags,
        "multimodal_embedding": embedding,
        "creator_trust_score": round(random.random(), 3) if not is_trojan else 0.1,
        "intensity": round(intens, 3),
        "drain": round(drain, 3),
        "connection": round(conn, 3),
        "growth": round(growth, 3),
        "age_appropriateness": round(age, 3)
    }

def build_content_database(size=5000, output_file="engine/content_db.json"):
    items = []
    for i in range(size):
        items.append(generate_content_item(i))
    with open(output_file, 'w') as f:
        json.dump(items, f, indent=2)
    return items

class ContentFactory:
    """Loads the pre-generated database to serve content candidates."""
    def __init__(self, db_path="engine/content_db.json"):
        if not Path(db_path).exists():
            print(f"Content DB not found at {db_path}. Generating now...")
            self.db = build_content_database(5000, db_path)
        else:
            with open(db_path, "r") as f:
                self.db = json.load(f)

    def get_candidates(self, count=5):
        """Returns `count` random content items from the database."""
        return random.sample(self.db, count)

    def compute_algorithmic_similarity(self, content_a, content_b):
        """Computes structural similarity natively using the multimodal latent embeddings."""
        if not content_a or not content_b:
             return 0.5
        v1 = content_a["multimodal_embedding"]
        v2 = content_b["multimodal_embedding"]
        # Convert [-1.0, 1.0] cosine to [0.0, 1.0] scale
        sim = (cosine_similarity(v1, v2) + 1.0) / 2.0
        return max(0.0, min(sim, 1.0))

