"""
Step 2: Change-Point Detection on Temporal Embeddings
=====================================================
This script loads the temporal embeddings from Step 1 and applies
change-point detection algorithms to automatically find events.
"""

import numpy as np
import json
import os
import ruptures as rpt
from sklearn.metrics.pairwise import cosine_similarity


def load_temporal_embeddings(output_dir: str, entity: str) -> dict:
    """Load saved embeddings for an entity."""
    embeddings = {}
    for f in sorted(os.listdir(output_dir)):
        if f.startswith(entity) and f.endswith(".npy"):
            period = f.replace(f"{entity}_", "").replace(".npy", "")
            embeddings[period] = np.load(os.path.join(output_dir, f))
    return embeddings


def compute_cosine_distance_series(embeddings: dict) -> tuple:
    """Compute cosine distance time series between consecutive periods."""
    sorted_periods = sorted(embeddings.keys())
    distances = []
    transitions = []
    
    for i in range(1, len(sorted_periods)):
        prev = embeddings[sorted_periods[i - 1]].reshape(1, -1)
        curr = embeddings[sorted_periods[i]].reshape(1, -1)
        dist = 1 - cosine_similarity(prev, curr)[0][0]
        distances.append(dist)
        transitions.append(f"{sorted_periods[i-1]} -> {sorted_periods[i]}")
    
    return np.array(distances), transitions, sorted_periods


def detect_change_points_pelt(signal: np.ndarray, penalty: float = 1.0) -> list:
    """
    Use PELT algorithm to detect change points.
    
    PELT = Pruned Exact Linear Time
    - Good for offline detection (you have all the data)
    - penalty controls sensitivity (lower = more change points detected)
    """
    if len(signal) < 3:
        print("Signal too short for PELT, using threshold method instead")
        mean = np.mean(signal)
        std = np.std(signal)
        return [i for i, v in enumerate(signal) if v > mean + std]
    
    algo = rpt.Pelt(model="rbf", min_size=1).fit(signal.reshape(-1, 1))
    change_points = algo.predict(pen=penalty)
    change_points = [cp for cp in change_points if cp < len(signal)]
    return change_points


def detect_change_points_threshold(signal: np.ndarray, n_std: float = 1.5) -> list:
    """
    Simple threshold-based detection.
    Flag any point that is more than n_std standard deviations above the mean.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    threshold = mean + n_std * std
    
    change_points = [i for i, v in enumerate(signal) if v > threshold]
    return change_points


def detect_change_points_binseg(signal: np.ndarray, n_bkps: int = 2) -> list:
    """
    Binary Segmentation algorithm.
    You specify how many breakpoints you expect.
    """
    if len(signal) < 5:
        return detect_change_points_threshold(signal)
    
    try:
        algo = rpt.Binseg(model="l2", min_size=1).fit(signal.reshape(-1, 1))
        change_points = algo.predict(n_bkps=min(n_bkps, len(signal) - 2))
        change_points = [cp for cp in change_points if cp < len(signal)]
        return change_points
    except Exception:
        return detect_change_points_threshold(signal)


if __name__ == "__main__":
    print("=" * 60)
    print("CHANGE-POINT DETECTION FOR EVENT DISCOVERY")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    
    if not os.path.exists(os.path.join(output_dir, "temporal_distances.json")):
        print("Run 01_extract_embeddings.py first!")
        exit(1)
    
    with open(os.path.join(output_dir, "temporal_distances.json")) as f:
        all_distances = json.load(f)
    
    for entity in all_distances:
        print(f"\n{'='*40}")
        print(f"Entity: {entity}")
        print(f"{'='*40}")
        
        embeddings = load_temporal_embeddings(output_dir, entity)
        distances, transitions, periods = compute_cosine_distance_series(embeddings)
        
        print(f"\nDistance time series:")
        for t, d in zip(transitions, distances):
            print(f"  {t}: {d:.6f}")
        
        print(f"\nMethod 1 - Threshold (1.5 std):")
        cp_threshold = detect_change_points_threshold(distances, n_std=1.0)
        if cp_threshold:
            for cp in cp_threshold:
                print(f"  Event detected at: {transitions[cp]} (distance: {distances[cp]:.6f})")
        else:
            print("  No events detected")
        
        print(f"\nMethod 2 - Binary Segmentation (n_bkps=1):")
        cp_binseg = detect_change_points_binseg(distances, n_bkps=1)
        if cp_binseg:
            for cp in cp_binseg:
                if cp < len(transitions):
                    print(f"  Event detected at: {transitions[cp]} (distance: {distances[cp]:.6f})")
        else:
            print("  No events detected")
        
        print(f"\nMethod 3 - PELT:")
        cp_pelt = detect_change_points_pelt(distances, penalty=0.01)
        if cp_pelt:
            for cp in cp_pelt:
                if cp < len(transitions):
                    print(f"  Event detected at: {transitions[cp]} (distance: {distances[cp]:.6f})")
        else:
            print("  No events detected")
        
        results = {
            "entity": entity,
            "transitions": transitions,
            "distances": distances.tolist(),
            "detected_events": {
                "threshold": [transitions[cp] for cp in cp_threshold if cp < len(transitions)],
                "binseg": [transitions[cp] for cp in cp_binseg if cp < len(transitions)],
                "pelt": [transitions[cp] for cp in cp_pelt if cp < len(transitions)],
            }
        }
        
        with open(os.path.join(output_dir, f"events_{entity}.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvent detection results saved to {output_dir}/")
    print("\nNEXT: Run 03_visualize.py to see the plots!")

