"""
Phase 10: Conceptual Continuity Analysis
Implements Topic Modeling, Semantic Clustering, and Global Coherence
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Load cached embeddings
print("Loading embeddings...")
with open('/Users/carlosdaniel/Documents/Blog/carlosjimenez88m.github.io/tidytuesday/data/embeddings_ada002_cache.pkl', 'rb') as f:
    df_lyrics_ada = pickle.load(f)

print(f"✓ Loaded {len(df_lyrics_ada)} lines with ada-002 embeddings")
print(f"Artists: {df_lyrics_ada['artist'].unique()}")

print("\n" + "="*80)
print("PHASE 10: CONCEPTUAL CONTINUITY ANALYSIS")
print("="*80)

# =============================================================================
# Method 5: Topic Modeling with LDA
# =============================================================================
print("\n### Method 5: Topic Modeling (LDA) ###\n")

def calculate_topic_persistence_lda(df, n_topics=5):
    """
    Use Latent Dirichlet Allocation to extract topics and measure persistence.
    """
    results = []

    for artist in ['Pink Floyd', 'The Beatles']:
        artist_df = df[df['artist'] == artist]

        for song, group in artist_df.groupby('song'):
            if len(group) < 5:
                continue

            docs = group['lyric_line'].tolist()

            # Vectorize lyrics
            try:
                vectorizer = CountVectorizer(
                    max_features=200,
                    stop_words='english',
                    min_df=1,
                    max_df=0.95
                )
                doc_term_matrix = vectorizer.fit_transform(docs)

                # LDA
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=20
                )
                topic_distributions = lda.fit_transform(doc_term_matrix)

                # Calculate topic persistence
                dominant_topics = topic_distributions.argmax(axis=1)

                # Count consecutive lines with same dominant topic
                windows = []
                for i in range(len(dominant_topics)):
                    window = 0
                    for j in range(i + 1, len(dominant_topics)):
                        if dominant_topics[j] == dominant_topics[i]:
                            window += 1
                        else:
                            break
                    windows.append(window)

                persistence = np.mean(windows)
                n_unique_topics = len(set(dominant_topics))

                results.append({
                    'artist': artist,
                    'song': song,
                    'topic_persistence': persistence,
                    'n_topics': n_unique_topics,
                    'n_lines': len(docs)
                })

            except Exception as e:
                print(f"  Skipping {song}: {str(e)[:50]}")
                continue

    return pd.DataFrame(results)

# Execute
print("Running LDA topic modeling...")
df_topics = calculate_topic_persistence_lda(df_lyrics_ada, n_topics=5)

print("\n=== Topic Modeling Results ===")
print(df_topics.groupby('artist')[['topic_persistence', 'n_topics']].describe())

# Statistical test
floyd_topics = df_topics[df_topics['artist'] == 'Pink Floyd']['topic_persistence']
beatles_topics = df_topics[df_topics['artist'] == 'The Beatles']['topic_persistence']

from scipy import stats
t_stat, p_value = stats.ttest_ind(floyd_topics, beatles_topics)

print(f"\nStatistical Test:")
print(f"Pink Floyd mean: {floyd_topics.mean():.3f} (SD: {floyd_topics.std():.3f})")
print(f"Beatles mean: {beatles_topics.mean():.3f} (SD: {beatles_topics.std():.3f})")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"Result: {'✓ Significant difference' if p_value < 0.05 else '✗ No significant difference'}")

# Save
df_topics.to_csv('/Users/carlosdaniel/Documents/Blog/carlosjimenez88m.github.io/tidytuesday/data/topic_modeling_results.csv', index=False)
print("\n✓ Results saved to data/topic_modeling_results.csv")

# =============================================================================
# Method 6: Semantic Clustering Analysis
# =============================================================================
print("\n\n### Method 6: Semantic Clustering (K-Means) ###\n")

def calculate_cluster_continuity(df, n_clusters=5):
    """
    Cluster embeddings using K-Means and measure cluster persistence.
    """
    results = []

    for artist in ['Pink Floyd', 'The Beatles']:
        artist_df = df[df['artist'] == artist]

        for song, group in artist_df.groupby('song'):
            if len(group) < 5:
                continue

            # Extract embeddings
            embeddings = np.array(group['embedding'].tolist())

            # K-Means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(group)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Calculate cluster continuity (consecutive lines in same cluster)
            windows = []
            for i in range(len(cluster_labels)):
                window = 0
                for j in range(i + 1, len(cluster_labels)):
                    if cluster_labels[j] == cluster_labels[i]:
                        window += 1
                    else:
                        break
                windows.append(window)

            continuity = np.mean(windows)
            n_unique_clusters = len(set(cluster_labels))

            # Calculate primary cluster distribution
            from collections import Counter
            cluster_dist = Counter(cluster_labels)
            primary_clusters = ", ".join([f"C{k}:{v}" for k, v in sorted(cluster_dist.items())])

            results.append({
                'artist': artist,
                'song': song,
                'cluster_continuity': continuity,
                'n_clusters': n_unique_clusters,
                'n_lines': len(group),
                'cluster_distribution': primary_clusters
            })

    return pd.DataFrame(results)

# Execute
print("Running K-Means semantic clustering...")
df_clusters = calculate_cluster_continuity(df_lyrics_ada, n_clusters=5)

print("\n=== Semantic Clustering Results ===")
print(df_clusters.groupby('artist')[['cluster_continuity', 'n_clusters']].describe())

# Statistical test
floyd_clusters = df_clusters[df_clusters['artist'] == 'Pink Floyd']['cluster_continuity']
beatles_clusters = df_clusters[df_clusters['artist'] == 'The Beatles']['cluster_continuity']

t_stat, p_value = stats.ttest_ind(floyd_clusters, beatles_clusters)

print(f"\nStatistical Test:")
print(f"Pink Floyd mean: {floyd_clusters.mean():.3f} (SD: {floyd_clusters.std():.3f})")
print(f"Beatles mean: {beatles_clusters.mean():.3f} (SD: {beatles_clusters.std():.3f})")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"Result: {'✓ Significant difference' if p_value < 0.05 else '✗ No significant difference'}")

# Save
df_clusters.to_csv('/Users/carlosdaniel/Documents/Blog/carlosjimenez88m.github.io/tidytuesday/data/semantic_clustering_results.csv', index=False)
print("\n✓ Results saved to data/semantic_clustering_results.csv")

# =============================================================================
# Method 7: Global Coherence (All-Pairs Similarity)
# =============================================================================
print("\n\n### Method 7: Global Coherence Analysis ###\n")

def calculate_global_coherence(df):
    """
    Calculate mean pairwise similarity between all lines in a song.
    Captures long-range semantic connections.
    """
    results = []

    for artist in ['Pink Floyd', 'The Beatles']:
        artist_df = df[df['artist'] == artist]

        for song, group in artist_df.groupby('song'):
            if len(group) < 3:
                continue

            # Extract embeddings
            embeddings = np.array(group['embedding'].tolist())

            # Calculate pairwise similarity matrix
            sim_matrix = cosine_similarity(embeddings)

            # Mean of off-diagonal elements (exclude self-similarity)
            mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
            global_coherence = sim_matrix[mask].mean()
            std_coherence = sim_matrix[mask].std()

            results.append({
                'artist': artist,
                'song': song,
                'global_coherence': global_coherence,
                'std_coherence': std_coherence,
                'n_lines': len(group)
            })

    return pd.DataFrame(results)

# Execute
print("Running global coherence analysis...")
df_global = calculate_global_coherence(df_lyrics_ada)

print("\n=== Global Coherence Results ===")
print(df_global.groupby('artist')[['global_coherence', 'std_coherence']].describe())

# Statistical test
floyd_global = df_global[df_global['artist'] == 'Pink Floyd']['global_coherence']
beatles_global = df_global[df_global['artist'] == 'The Beatles']['global_coherence']

t_stat, p_value = stats.ttest_ind(floyd_global, beatles_global)

print(f"\nStatistical Test:")
print(f"Pink Floyd mean: {floyd_global.mean():.4f} (SD: {floyd_global.std():.4f})")
print(f"Beatles mean: {beatles_global.mean():.4f} (SD: {beatles_global.std():.4f})")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"Result: {'✓ Significant difference' if p_value < 0.05 else '✗ No significant difference'}")

# Save
df_global.to_csv('/Users/carlosdaniel/Documents/Blog/carlosjimenez88m.github.io/tidytuesday/data/global_coherence_results.csv', index=False)
print("\n✓ Results saved to data/global_coherence_results.csv")

# =============================================================================
# Summary Comparison
# =============================================================================
print("\n\n" + "="*80)
print("CONCEPTUAL CONTINUITY SUMMARY")
print("="*80)

summary_data = {
    'Metric': ['Topic Persistence (LDA)', 'Cluster Continuity (K-Means)', 'Global Coherence (All-pairs)'],
    'Pink Floyd Mean': [
        floyd_topics.mean(),
        floyd_clusters.mean(),
        floyd_global.mean()
    ],
    'Beatles Mean': [
        beatles_topics.mean(),
        beatles_clusters.mean(),
        beatles_global.mean()
    ],
    'Ratio (Floyd/Beatles)': [
        floyd_topics.mean() / beatles_topics.mean() if beatles_topics.mean() > 0 else 0,
        floyd_clusters.mean() / beatles_clusters.mean() if beatles_clusters.mean() > 0 else 0,
        floyd_global.mean() / beatles_global.mean() if beatles_global.mean() > 0 else 0
    ]
}

df_summary = pd.DataFrame(summary_data)
print("\n")
print(df_summary.to_string(index=False))

# Check if hypothesis is supported
print("\n" + "="*80)
print("HYPOTHESIS EVALUATION:")
print("="*80)

if floyd_topics.mean() > beatles_topics.mean():
    print("✓ Topic Persistence: Floyd > Beatles (supports hypothesis)")
else:
    print("✗ Topic Persistence: Floyd ≤ Beatles (contradicts hypothesis)")

if floyd_clusters.mean() > beatles_clusters.mean():
    print("✓ Cluster Continuity: Floyd > Beatles (supports hypothesis)")
else:
    print("✗ Cluster Continuity: Floyd ≤ Beatles (contradicts hypothesis)")

if floyd_global.mean() > beatles_global.mean():
    print("✓ Global Coherence: Floyd > Beatles (supports hypothesis)")
else:
    print("✗ Global Coherence: Floyd ≤ Beatles (contradicts hypothesis)")

print("\n✓ Conceptual continuity analysis complete!")
print("="*80)
