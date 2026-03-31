"""
Step 3: Visualize Temporal Embeddings & Detected Events
=======================================================
Creates publication-quality plots of the embedding evolution and detected events.
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def plot_cosine_distances(output_dir: str, entity: str):
    """Plot cosine distance time series with detected events marked."""
    events_file = os.path.join(output_dir, f"events_{entity}.json")
    if not os.path.exists(events_file):
        print(f"No events file for {entity}")
        return
    
    with open(events_file) as f:
        results = json.load(f)
    
    transitions = results["transitions"]
    distances = results["distances"]
    detected = results["detected_events"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(distances))
    ax.plot(x, distances, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Cosine distance')
    ax.fill_between(x, distances, alpha=0.15, color='#2196F3')
    
    for event_transition in detected.get("threshold", []):
        if event_transition in transitions:
            idx = transitions.index(event_transition)
            ax.axvline(x=idx, color='#F44336', linestyle='--', alpha=0.7, linewidth=2)
            ax.scatter([idx], [distances[idx]], color='#F44336', s=200, zorder=5,
                      marker='*', label='Detected event' if event_transition == detected["threshold"][0] else "")
    
    mean_d = np.mean(distances)
    std_d = np.std(distances)
    ax.axhline(y=mean_d, color='gray', linestyle=':', alpha=0.5, label=f'Mean ({mean_d:.4f})')
    ax.axhline(y=mean_d + std_d, color='orange', linestyle=':', alpha=0.5, label=f'Mean + 1s ({mean_d + std_d:.4f})')
    
    ax.set_xticks(x)
    ax.set_xticklabels(transitions, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Cosine Distance', fontsize=12)
    ax.set_title(f'Temporal Embedding Distance - {entity}\n(Higher distance = bigger contextual change = potential event)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"plot_distances_{entity}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_self_similarity_matrix(output_dir: str, entity: str):
    """Create a self-similarity matrix showing entity embedding similarity across all time periods."""
    embeddings = {}
    for f in sorted(os.listdir(output_dir)):
        if f.startswith(entity) and f.endswith(".npy"):
            period = f.replace(f"{entity}_", "").replace(".npy", "")
            embeddings[period] = np.load(os.path.join(output_dir, f))
    
    if not embeddings:
        print(f"No embeddings found for {entity}")
        return
    
    sorted_periods = sorted(embeddings.keys())
    n = len(sorted_periods)
    
    sim_matrix = np.zeros((n, n))
    emb_array = np.array([embeddings[p] for p in sorted_periods])
    sim_matrix = cosine_similarity(emb_array)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        xticklabels=sorted_periods,
        yticklabels=sorted_periods,
        ax=ax,
        vmin=sim_matrix.min() - 0.001,
        vmax=1.0,
        square=True,
        linewidths=0.5
    )
    ax.set_title(f'Self-Similarity Matrix - {entity}\n(Green = similar contexts, Red = different contexts)',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"plot_similarity_{entity}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_embedding_trajectory(output_dir: str, entity: str):
    """Project high-dimensional embeddings to 2D using PCA to visualize trajectory over time."""
    embeddings = {}
    for f in sorted(os.listdir(output_dir)):
        if f.startswith(entity) and f.endswith(".npy"):
            period = f.replace(f"{entity}_", "").replace(".npy", "")
            embeddings[period] = np.load(os.path.join(output_dir, f))
    
    if len(embeddings) < 2:
        print(f"Need at least 2 time periods for trajectory plot")
        return
    
    sorted_periods = sorted(embeddings.keys())
    emb_array = np.array([embeddings[p] for p in sorted_periods])
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(emb_array)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(sorted_periods)))
    
    ax.plot(coords[:, 0], coords[:, 1], '-', color='gray', alpha=0.4, linewidth=1)
    
    for i in range(len(coords) - 1):
        dx = coords[i+1, 0] - coords[i, 0]
        dy = coords[i+1, 1] - coords[i, 1]
        ax.annotate('', xy=(coords[i+1, 0], coords[i+1, 1]),
                    xytext=(coords[i, 0], coords[i, 1]),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2, alpha=0.7))
    
    for i, (period, color) in enumerate(zip(sorted_periods, colors)):
        ax.scatter(coords[i, 0], coords[i, 1], c=[color], s=150, zorder=5, edgecolors='black', linewidth=1)
        ax.annotate(period, (coords[i, 0], coords[i, 1]),
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'Embedding Trajectory - {entity}\n(How the entity representation moves through semantic space over time)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"plot_trajectory_{entity}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZING TEMPORAL EMBEDDING ANALYSIS")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    
    entities = set()
    for f in os.listdir(output_dir):
        if f.endswith(".npy"):
            entity = f.rsplit("_", 1)[0]
            entities.add(entity)
    
    if not entities:
        print("No embeddings found! Run 01_extract_embeddings.py first.")
        exit(1)
    
    for entity in sorted(entities):
        print(f"\nGenerating plots for: {entity}")
        plot_cosine_distances(output_dir, entity)
        plot_self_similarity_matrix(output_dir, entity)
        plot_embedding_trajectory(output_dir, entity)
    
    print(f"\nAll plots saved to {output_dir}/")
    print("Open the PNG files to see the results!")
