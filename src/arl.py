"""
Emotional Centroid System - Complete Implementation
Combines keyword-based emotional engine with semantic centroid scoring
Based on Panksepp's affective neuroscience + Russell's circumplex model
"""

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import json
import re
from datetime import datetime


# ============================================================================
# ANCHOR SENTENCES FOR PANKSEPP'S SEEKING SYSTEM
# ============================================================================

SEEKING_ANCHORS = [
    "What happens if I mix these two chemicals together?",
    "I wonder what's behind that closed door at the end of the hallway.",
    "How does the brain generate consciousness from electrical signals?",
    "Let's explore the forest path we've never taken before.",
    "I'm curious about how ancient civilizations built those massive structures.",
    "What would happen if we tried a completely different approach to this problem?",
    "I want to understand why the stars appear to move across the sky.",
    "There's something interesting about this pattern I need to investigate further.",
    "How do birds know exactly when to migrate thousands of miles?",
    "I'm fascinated by what lies at the bottom of the deepest ocean trenches.",
    "What if we could see the world through the eyes of other animals?",
    "I need to figure out how this old machine works by taking it apart.",
    "Why do some songs give us chills while others don't?",
    "Let's see what happens when we change just one variable in the experiment.",
    "I'm intrigued by the possibility of undiscovered species in remote jungles.",
    "What causes some people to be more creative than others?",
    "I wonder what my ancestors were doing exactly 500 years ago today.",
    "How can we detect planets orbiting stars light-years away?",
    "There must be a reason why this keeps happening—I need to find out.",
    "What new discoveries await us if we keep pushing the boundaries of knowledge?"
]

JOY_ANCHORS = [
    "I'm so happy I could dance!",
    "This is the best day of my life!",
    "I feel absolutely wonderful right now.",
    "Everything is going perfectly today.",
    "I'm filled with pure joy and excitement.",
    "This makes me smile from ear to ear.",
    "I feel so alive and energized!",
    "My heart is overflowing with happiness.",
    "I'm having the time of my life!",
    "This brings me such immense pleasure.",
    "I'm thrilled beyond words!",
    "Life feels magical right now.",
    "I'm bursting with positive energy.",
    "This moment is absolutely perfect.",
    "I feel like I'm floating on air.",
    "My spirits are soaring high!",
    "This fills me with pure delight.",
    "I'm radiating happiness right now.",
    "Everything feels bright and beautiful.",
    "I'm grateful for this amazing feeling."
]

SADNESS_ANCHORS = [
    "I feel so alone and empty inside.",
    "My heart is heavy with sorrow.",
    "Nothing seems to matter anymore.",
    "I miss them so much it hurts.",
    "Everything feels gray and meaningless.",
    "I can't stop crying about what happened.",
    "The pain of this loss is overwhelming.",
    "I feel disconnected from everything.",
    "Why does everything have to end?",
    "I'm drowning in grief and despair.",
    "This sadness won't go away.",
    "I feel broken and defeated.",
    "The weight of this sadness is crushing me.",
    "I wish things could go back to how they were.",
    "I feel so helpless and hopeless.",
    "My world has lost all its color.",
    "I'm struggling to find meaning in anything.",
    "This emptiness inside me is unbearable.",
    "I feel like I'm falling into darkness.",
    "The sadness is too much to carry."
]

CREATIVITY_ANCHORS = [
    "What if we reimagine this in a completely different way?",
    "I have an idea that combines art and science in a new form.",
    "Let's create something that's never been done before.",
    "I'm thinking outside the box on this one.",
    "What if we flip all the rules upside down?",
    "I'm inspired to make something truly original.",
    "Let's experiment with unconventional approaches.",
    "I want to express this idea through multiple mediums.",
    "What happens when we merge these two unrelated concepts?",
    "I'm envisioning a fresh perspective on this problem.",
    "Let's play with possibilities and see what emerges.",
    "I'm imagining something wild and innovative.",
    "What if we break this down and rebuild it differently?",
    "I want to craft something unique and meaningful.",
    "Let's explore the creative potential here.",
    "I'm feeling inspired to innovate and experiment.",
    "What if we design this from a completely new angle?",
    "I'm driven to create something extraordinary.",
    "Let's push the boundaries of what's possible.",
    "I'm brainstorming radical new solutions."
]


# ============================================================================
# KEYWORD SETS 
# ============================================================================

HAPPY = [
    "happy", "joy", "excited", "love", "wonderful", "great", "amazing",
    "fantastic", "delighted", "cheerful", "pleased", "glad", "thrilled"
]

SAD = [
    "sad", "unhappy", "depressed", "miserable", "grief", "sorrow",
    "heartbroken", "devastated", "lonely", "empty", "crying", "tears"
]

CURIOUS = [
    "curious", "wonder", "why", "how", "what if", "explore", "discover",
    "investigate", "question", "intrigued", "fascinated", "interested"
]

CREATIVE = [
    "create", "imagine", "design", "innovative", "artistic", "original",
    "inventive", "craft", "build", "compose", "brainstorm", "envision"
]

GENERAL = [
    "think", "believe", "know", "understand", "consider", "seem", "appear"
]


# ============================================================================
# CENTROID GENERATOR
# ============================================================================

class EmotionalCentroidGenerator:
    """
    Generates semantic centroids for emotional categories using SentenceTransformers
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model identifier
                       'all-MiniLM-L6-v2' (fast, 384 dims)
                       'all-mpnet-base-v2' (better quality, 768 dims)
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def compute_centroid(self, sentences: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Compute the mean embedding (centroid) of a list of sentences.
        
        Args:
            sentences: List of anchor sentences
            normalize: Whether to L2-normalize the final centroid
            
        Returns:
            Centroid vector as PyTorch tensor
        """
        print(f"Encoding {len(sentences)} sentences...")
        
        # Encode all sentences
        embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Compute mean (centroid)
        centroid = torch.mean(embeddings, dim=0)
        
        # L2 normalization (beneficial for cosine similarity)
        if normalize:
            centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)
        
        return centroid
    
    def analyze_centroid(self, centroid: torch.Tensor, sentences: List[str]) -> Dict:
        """
        Analyze the centroid's relationship to its anchor sentences
        """
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        centroid_norm = torch.nn.functional.normalize(centroid.unsqueeze(0), p=2, dim=1)
        
        # Cosine similarities
        similarities = torch.mm(embeddings_norm, centroid_norm.T).squeeze()
        
        return {
            "mean_similarity": float(similarities.mean()),
            "std_similarity": float(similarities.std()),
            "min_similarity": float(similarities.min()),
            "max_similarity": float(similarities.max()),
            "similarities": similarities.cpu().numpy().tolist()
        }
    
    def save_centroid(self, centroid: torch.Tensor, filepath: str, metadata: Dict = None):
        """
        Save centroid to disk with optional metadata
        """
        save_dict = {
            "centroid": centroid.cpu().numpy().tolist(),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model._model_card_vars.get('model_name', 'unknown'),
            "metadata": metadata or {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)
        print(f"Centroid saved to: {filepath}")
    
    def load_centroid(self, filepath: str) -> Tuple[torch.Tensor, Dict]:
        """
        Load centroid from disk
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        centroid = torch.tensor(data["centroid"], dtype=torch.float32)
        print(f"Loaded centroid: {data['embedding_dim']} dims from {data['model_name']}")
        
        return centroid, data.get("metadata", {})


# ============================================================================
# EMOTIONAL ENGINE (Keyword + Semantic)
# ============================================================================

class EmotionalEngine:
    """
    Combines keyword-based scoring with semantic centroid similarity.
    Scientifically grounded in:
       - Russell's circumplex model (Valence x Arousal)
       - Panksepp's 7 core affective systems
       - Plutchik's wheel of emotions
       - Semantic similarity via transformer embeddings
    """

    def __init__(self, use_semantic: bool = True, semantic_weight: float = 0.5):
        """
        Args:
            use_semantic: Whether to use semantic scoring
            semantic_weight: Weight for semantic scores (0.0-1.0)
                           0.0 = pure keyword, 1.0 = pure semantic
        """
        # Current emotional state
        self.valence = 0.0  # -1.0 (very negative) -> +1.0 (very positive)
        self.arousal = 0.0  # 0.0 (calm/asleep) -> 1.0 (highly activated)

        # Primary affective drives
        self.joy = 0.0
        self.sadness = 0.0
        self.curiosity = 0.0
        self.creativity = 0.0
        self.fear = 0.0
        self.anger = 0.0

        # Keyword sets
        self.triggers = {
            "joy": HAPPY,
            "sadness": SAD,
            "curiosity": CURIOUS,
            "creativity": CREATIVE,
            "neutral": GENERAL
        }
        
        # Semantic scoring setup
        self.use_semantic = use_semantic
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        
        self.centroids = {}
        self.centroid_generator = None
        
        if use_semantic:
            print("\nInitializing semantic scoring system...")
            self._initialize_semantic_scoring()

    def _initialize_semantic_scoring(self):
        """Load or compute emotional centroids"""
        self.centroid_generator = EmotionalCentroidGenerator()
        
        # Compute centroids for each emotion
        print("\nComputing emotional centroids...")
        
        centroid_data = {
            "joy": JOY_ANCHORS,
            "sadness": SADNESS_ANCHORS,
            "curiosity": SEEKING_ANCHORS,
            "creativity": CREATIVITY_ANCHORS
        }
        
        for emotion, anchors in centroid_data.items():
            print(f"\n  → {emotion.capitalize()}")
            centroid = self.centroid_generator.compute_centroid(anchors, normalize=True)
            self.centroids[emotion] = centroid
            
            # Optional: analyze quality
            analysis = self.centroid_generator.analyze_centroid(centroid, anchors)
            print(f"     Mean similarity: {analysis['mean_similarity']:.4f}")
        
        print("\n✓ Semantic scoring ready!\n")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())

    def _keyword_score(self, text: str) -> Dict[str, float]:
        """Original keyword-based scoring"""
        tokens = self._tokenize(text)
        text_str = " " + " ".join(tokens) + " "

        scores = {
            "joy": 0.0,
            "sadness": 0.0,
            "curiosity": 0.0,
            "creativity": 0.0
        }

        # Count keyword matches
        for emotion, keywords in self.triggers.items():
            if emotion == "neutral":
                continue
            matches = sum(1 for kw in keywords if kw in text_str)
            if matches > 0:
                intensity = min(matches * 0.35, 1.0)
                if emotion == "joy":
                    scores["joy"] = max(scores["joy"], intensity * 0.95)
                elif emotion == "sadness":
                    scores["sadness"] = max(scores["sadness"], intensity * 0.90)
                elif emotion == "curiosity":
                    scores["curiosity"] = max(scores["curiosity"], intensity * 0.85)
                elif emotion == "creativity":
                    scores["creativity"] = max(scores["creativity"], intensity * 0.88)

        return scores

    def _semantic_score(self, text: str) -> Dict[str, float]:
        """Semantic similarity scoring using centroids"""
        if not self.use_semantic or not self.centroid_generator:
            return {"joy": 0.0, "sadness": 0.0, "curiosity": 0.0, "creativity": 0.0}
        
        # Encode input text
        text_embedding = self.centroid_generator.model.encode(text, convert_to_tensor=True)
        text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=0)
        
        scores = {}
        for emotion, centroid in self.centroids.items():
            centroid_norm = torch.nn.functional.normalize(centroid, p=2, dim=0)
            similarity = torch.dot(text_embedding, centroid_norm).item()
            
            # Map similarity from [-1, 1] to [0, 1]
            score = (similarity + 1.0) / 2.0
            scores[emotion] = score
        
        return scores

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Returns emotional profile for input text.
        Combines keyword and semantic scoring.
        """
        # Get both scoring types
        keyword_scores = self._keyword_score(text)
        semantic_scores = self._semantic_score(text)
        
        # Combine scores
        scores = {}
        for emotion in ["joy", "sadness", "curiosity", "creativity"]:
            keyword_val = keyword_scores.get(emotion, 0.0)
            semantic_val = semantic_scores.get(emotion, 0.0)
            
            # Weighted combination
            combined = (self.keyword_weight * keyword_val + 
                       self.semantic_weight * semantic_val)
            scores[emotion] = combined

        # Biological cross-modulation
        if scores["joy"] > 0.4:
            scores["curiosity"] += scores["joy"] * 0.4
            scores["creativity"] += scores["joy"] * 0.45

        if scores["sadness"] > 0.4:
            scores["curiosity"] -= scores["sadness"] * 0.5
            scores["creativity"] -= scores["sadness"] * 0.4

        if scores["curiosity"] > 0.6:
            scores["creativity"] += 0.2

        # Valence and arousal (Russell's circumplex)
        self.valence = scores["joy"] * 1.0 - scores["sadness"] * 1.0
        self.arousal = (scores["joy"] * 0.7 +
                       scores["curiosity"] * 0.8 +
                       scores["creativity"] * 0.6 +
                       scores["sadness"] * 0.5)

        # Clamp
        self.valence = max(min(self.valence, 1.0), -1.0)
        self.arousal = max(min(self.arousal, 1.0), 0.0)

        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "joy": round(scores["joy"], 3),
            "sadness": round(scores["sadness"], 3),
            "curiosity": round(scores["curiosity"], 3),
            "creativity": round(scores["creativity"], 3),
            "dominant": self._get_dominant_emotion(scores),
            "keyword_scores": keyword_scores,
            "semantic_scores": semantic_scores
        }

    def _get_dominant_emotion(self, scores: dict) -> str:
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ordered[0][0] if ordered[0][1] > 0.3 else "neutral"


# ============================================================================
# HOMEOSTATIC EMOTIONAL SYSTEM
# ============================================================================

class PioneerEmotionalSystem:
    """
    Biologically-inspired homeostasis with real-time decay.
    Integrates both keyword and semantic emotional scoring.
    """

    def __init__(self,
                 half_life_seconds: float = 15.0,
                 joy_decay: float = 1.0,
                 sad_decay: float = 1.2,
                 curiosity_decay: float = 0.9,
                 creative_decay: float = 0.95,
                 use_semantic: bool = True,
                 semantic_weight: float = 0.5):
        
        self.engine = EmotionalEngine(
            use_semantic=use_semantic,
            semantic_weight=semantic_weight
        )
        self.last_update = datetime.now()

        # Individual half-lives (biology: sadness lingers longer than joy)
        self.decay_rates = torch.tensor([
            joy_decay / half_life_seconds,
            sad_decay / half_life_seconds,
            curiosity_decay / half_life_seconds,
            creative_decay / half_life_seconds
        ], dtype=torch.float32)

        # Persistent emotional state
        self.emotion_state = torch.zeros(4, dtype=torch.float32)

    def process(self, text: str) -> Dict[str, float]:
        """
        Full emotional life cycle: decay old emotions + integrate new input
        """
        now = datetime.now()
        delta = (now - self.last_update).total_seconds()
        self.last_update = now

        # 1. Homeostasis: decay old emotions
        if delta > 0:
            decay_factor = torch.exp(-self.decay_rates * delta)
            self.emotion_state *= decay_factor

        # 2. New emotional pulse
        scores = self.engine.score_text(text)
        new_pulse = torch.tensor([
            scores["joy"],
            scores["sadness"],
            scores["curiosity"],
            scores["creativity"]
        ], dtype=torch.float32)

        # 3. Integrate into living state
        self.emotion_state += new_pulse
        self.emotion_state = torch.clamp(self.emotion_state, -1.0, 1.0)

        # 4. Return current feeling
        return {
            "valence": round(float(self.engine.valence), 3),
            "arousal": round(float(self.engine.arousal), 3),
            "joy": round(float(self.emotion_state[0]), 3),
            "sadness": round(float(self.emotion_state[1]), 3),
            "curiosity": round(float(self.emotion_state[2]), 3),
            "creativity": round(float(self.emotion_state[3]), 3),
            "dominant": scores["dominant"],
            "raw_vector": self.emotion_state.clone(),
            "keyword_contribution": scores.get("keyword_scores", {}),
            "semantic_contribution": scores.get("semantic_scores", {})
        }

    def reset(self):
        """Reset emotional state to baseline"""
        self.emotion_state.zero_()
        self.last_update = datetime.now()


# ============================================================================
# DEMO / TESTING
# ============================================================================

def demo():
    """
    Demonstration of the complete emotional system
    """
    print("="*70)
    print("EMOTIONAL SYSTEM DEMONSTRATION")
    print("="*70)
    
    # Initialize system with semantic scoring
    system = PioneerEmotionalSystem(
        half_life_seconds=15.0,
        use_semantic=True,
        semantic_weight=0.5  # 50% keyword, 50% semantic
    )
    
    # Test sentences
    test_inputs = [
        "I'm so happy right now!",
        "This is the best day of my life!",
        "I feel absolutely wonderful and content.",
        "I'm thrilled about this news!",
        "Everything is going perfectly today.",
        "I'm filled with happiness and gratitude.",
        "This makes me so incredibly happy.",
        "I feel blessed and joyful.",
        "I'm delighted by this outcome.",
        "My heart is full of joy.",
        # Implicit
        "Everything just clicked into place.",
        "I can't stop smiling.",
        "Life is really good right now.",
        "Things are finally working out.",
        "I feel like I'm floating.",
        "The sun seems brighter today.",
        "I could dance right now.",
        "Everything feels right with the world.",
        "I'm in such a good place mentally.",
        "This feeling is amazing.",
        # Subtle
        "Things are okay, better than usual actually.",
        "I'm satisfied with how this turned out.",
        "That went well.",
        "I'm pleased with the results.",
        "Not bad at all, quite good actually.",
        "I'm so sad and heartbroken.",
        "I feel miserable and alone.",
        "Everything feels hopeless right now.",
        "I'm crying and can't stop.",
        "My heart aches with grief.",
        "I feel empty and lost inside.",
        "The pain is overwhelming.",
        "I'm devastated by this loss.",
        "Nothing matters anymore.",
        "I feel hollow and numb.",
        # Implicit
        "I don't want to get out of bed.",
        "Everything seems gray and pointless.",
        "I just want to be alone.",
        "I can't find joy in anything anymore.",
        "The days all blur together.",
        "I'm tired of pretending I'm okay.",
        "Nothing feels worth the effort.",
        "I miss how things used to be.",
        "Why does everything have to end?",
        "I feel disconnected from everyone.",
        # Subtle
        "I'm not doing great today.",
        "Things could be better honestly.",
        "I'm feeling a bit down.",
        "Today's been rough.",
        "I'm struggling a little.",
        "How does this actually work?",
        "I wonder why this happens.",
        "What causes this phenomenon?",
        "Can you explain this to me?",
        "I'm curious about the mechanism behind this.",
        "Why does this occur?",
        "I don't understand this yet.",
        "What's the explanation for this?",
        "I need to learn more about this topic.",
        "How do they accomplish this?",
        # Implicit
        "I wonder what lies beyond the horizon.",
        "There's something fascinating about this mystery.",
        "I need to investigate this further.",
        "What secrets does this hold?",
        "This puzzle needs solving.",
        "I'm intrigued by this pattern.",
        "There must be more to this story.",
        "What if there's something we're missing?",
        "This deserves deeper examination.",
        "I sense there's an explanation here.",
        # Subtle
        "That's interesting.",
        "I'd like to know more.",
        "Tell me about that.",
        "What do you mean by that?",
        "Could you elaborate?",
        "I'm going to design something unique.",
        "Let me create an original solution.",
        "I'll invent a new approach to this.",
        "I want to build something innovative.",
        "I'm developing a creative concept.",
        "Let me craft something original.",
        "I'll compose a new piece.",
        "I'm designing this from scratch.",
        "I want to express this artistically.",
        "Let me generate some creative ideas.",
        # Implicit
        "What if we flip this completely upside down?",
        "I'm imagining something totally different.",
        "Let's try a radically new angle.",
        "I see this in a completely fresh way.",
        "There's an unconventional solution here.",
        "I'm envisioning something unprecedented.",
        "This calls for outside-the-box thinking.",
        "I'm sketching out an original vision.",
        "Let's break all the usual rules.",
        "I'm thinking in entirely new directions.",
        # Subtle
        "Maybe we could try something different.",
        "I have an idea for this.",
        "There might be another way.",
        "Let me think of alternatives.",
        "What about a new approach?",
    ]
    
    print("\n" + "="*70)
    print("PROCESSING TEST INPUTS")
    print("="*70)
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\n[{i}] Input: \"{text}\"")
        result = system.process(text)
        
        print(f"    Dominant: {result['dominant']}")
        print(f"    Valence: {result['valence']:+.3f}  |  Arousal: {result['arousal']:.3f}")
        print(f"    Emotions: Joy={result['joy']:.3f}, Sad={result['sadness']:.3f}, " +
              f"Curious={result['curiosity']:.3f}, Creative={result['creativity']:.3f}")
        
        if result.get('keyword_contribution'):
            print(f"    [Keyword]: {result['keyword_contribution']}")
        if result.get('semantic_contribution'):
            print(f"    [Semantic]: {result['semantic_contribution']}")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    demo()
