"""
Step 1: Extract Contextual Embeddings for Event Detection
=========================================================
Extract contextual word embeddings for named entities (political figures)
from French text and track their evolution over time.

Uses distilbert-multilingual as baseline (works on Mac ARM).
When you have GPU access at the lab, switch to CamemBERT for better results.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List
import json
import gc


# ============================================================
# 1. LOAD MODEL
# ============================================================

def load_model(model_name: str = "distilbert/distilbert-base-multilingual-cased"):
    """
    Load transformer model.
    
    Models to try (when you have GPU):
    - "distilbert/distilbert-base-multilingual-cased"  <- works on Mac now
    - "almanach/camembert-base"  <- best French model (use at lab with GPU)
    - "flaubert/flaubert_base_cased"  <- alternative French model
    - "bert-base-multilingual-cased"  <- multilingual baseline
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    print(f"Model loaded! Hidden size: {model.config.hidden_size}")
    return tokenizer, model


# ============================================================
# 2. EXTRACT ENTITY EMBEDDING
# ============================================================

def get_entity_embedding(text, entity, tokenizer, model, layer=-1, pooling="mean"):
    """Extract contextual embedding of a named entity in a sentence."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[layer]
    
    # Find entity token positions
    entity_tokens = tokenizer.encode(entity, add_special_tokens=False)
    input_ids = inputs["input_ids"][0].tolist()
    
    entity_positions = []
    for i in range(len(input_ids) - len(entity_tokens) + 1):
        if input_ids[i:i + len(entity_tokens)] == entity_tokens:
            entity_positions = list(range(i, i + len(entity_tokens)))
            break
    
    if not entity_positions:
        entity_positions = [0]  # fallback: CLS token
    
    entity_emb = hidden_states[0, entity_positions, :]
    
    if pooling == "mean":
        result = entity_emb.mean(dim=0)
    elif pooling == "first":
        result = entity_emb[0]
    elif pooling == "max":
        result = entity_emb.max(dim=0).values
    
    out = result.detach().cpu().numpy().copy()
    del outputs, hidden_states, entity_emb, result, inputs
    gc.collect()
    return out


# ============================================================
# 3. SAMPLE TEMPORAL DATA (replace with INA data later)
# ============================================================

def create_sample_temporal_data() -> Dict[str, Dict[str, List[str]]]:
    """
    Simulated French news about political figures across time periods.
    
    The KEY idea: during period 2024-03, Macron's context changes
    dramatically (scandal), which should create a RUPTURE in his
    embedding vector - this is what we want to detect!
    """
    return {
        "Macron": {
            "2024-01": [
                "Macron a preside le conseil des ministres ce matin.",
                "Le president Macron a rencontre le chancelier allemand a Berlin.",
                "Macron a defendu sa reforme devant l'Assemblee nationale.",
            ],
            "2024-02": [
                "Macron s'est rendu en Afrique pour une tournee diplomatique.",
                "Le president Macron a inaugure un nouveau centre hospitalier.",
                "Macron a prononce un discours sur la transition energetique.",
            ],
            "2024-03": [
                "Macron est au coeur d'un scandale politique sans precedent.",
                "La cote de popularite de Macron s'effondre apres les revelations.",
                "Macron fait face a une motion de censure apres la crise.",
            ],
            "2024-04": [
                "Macron tente de reprendre la main apres la crise politique.",
                "Le president Macron a annonce un remaniement ministeriel.",
                "Macron cherche a apaiser les tensions apres des semaines de crise.",
            ],
            "2024-05": [
                "Macron a accueilli les chefs d'Etat europeens pour un sommet.",
                "Le president Macron a signe un accord commercial avec le Japon.",
                "Macron a visite une usine de semiconducteurs dans le sud de la France.",
            ],
        }
    }


# ============================================================
# 4. BUILD TEMPORAL EMBEDDINGS
# ============================================================

def build_temporal_embeddings(data, tokenizer, model, layer=-1, pooling="mean"):
    """For each entity + time period, compute averaged embedding."""
    results = {}
    
    for entity, periods in data.items():
        results[entity] = {}
        print(f"\nProcessing: {entity}")
        
        for period, sentences in periods.items():
            embeddings = []
            for sentence in sentences:
                emb = get_entity_embedding(sentence, entity, tokenizer, model, layer, pooling)
                embeddings.append(emb)
            
            avg = np.mean(embeddings, axis=0)
            results[entity][period] = avg
            print(f"  {period}: {len(embeddings)} sentences -> embedding shape {avg.shape}")
    
    return results


# ============================================================
# 5. COMPUTE TEMPORAL DISTANCES
# ============================================================

def compute_temporal_distances(temporal_embeddings):
    """Compute cosine distance between consecutive time periods."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    distances = {}
    for entity, periods in temporal_embeddings.items():
        sorted_p = sorted(periods.keys())
        distances[entity] = {}
        
        print(f"\nCosine distances for {entity}:")
        for i in range(1, len(sorted_p)):
            prev, curr = sorted_p[i-1], sorted_p[i]
            prev_emb = periods[prev].reshape(1, -1)
            curr_emb = periods[curr].reshape(1, -1)
            
            dist = float(1 - cosine_similarity(prev_emb, curr_emb)[0][0])
            transition = f"{prev} -> {curr}"
            distances[entity][transition] = dist
            
            flag = " POTENTIAL EVENT!" if dist > np.mean(list(distances[entity].values())) * 1.3 else ""
            print(f"  {transition}: {dist:.6f}{flag}")
    
    return distances


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DIACHRONIC EMBEDDING ANALYSIS FOR EVENT DETECTION")
    print("=" * 60)
    
    tokenizer, model = load_model()
    
    print("\nCreating sample temporal data...")
    data = create_sample_temporal_data()
    
    print("\nExtracting temporal embeddings...")
    temporal_embeddings = build_temporal_embeddings(data, tokenizer, model)
    
    print("\nComputing temporal distances...")
    distances = compute_temporal_distances(temporal_embeddings)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "temporal_distances.json"), "w") as f:
        json.dump(distances, f, indent=2, ensure_ascii=False)
    
    for entity, periods in temporal_embeddings.items():
        for period, emb in periods.items():
            np.save(os.path.join(output_dir, f"{entity}_{period}.npy"), emb)
    
    print(f"\nAll results saved to {output_dir}/")
    print("\nNEXT: python src/02_change_point_detection.py")
    print("      python src/03_visualize.py")
