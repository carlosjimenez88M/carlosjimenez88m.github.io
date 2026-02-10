#!/usr/bin/env python3
"""
Extended Analysis Script: OpenAI ada-002 + Multi-threshold + BERTopic + STS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Load environment variables
load_dotenv('/Users/carlosdaniel/Documents/Blog/.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

print("=" * 60)
print("EXTENDED ATTENTION WINDOWS ANALYSIS")
print("=" * 60)

# Helper function
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calculate_attention_window(song_df, threshold=0.70):
    """Calculate attention window for each line in a song."""
    embeddings = np.array(song_df['embedding'].tolist())
    windows = []

    for i in range(len(embeddings)):
        base_emb = embeddings[i]
        window_size = 0

        for j in range(i + 1, len(embeddings)):
            similarity = cosine_sim(base_emb, embeddings[j])
            if similarity > threshold:
                window_size += 1
            else:
                break

        windows.append(window_size)

    return windows

# ========================================
# PHASE 1: OpenAI ada-002 Embeddings
# ========================================
print("\n" + "=" * 60)
print("PHASE 1: OpenAI ada-002 Embeddings (1536-dim)")
print("=" * 60)

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding_ada002(text, model="text-embedding-ada-002"):
    """Get 1536-dim embeddings with OpenAI ada-002"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

embeddings_ada002_cache_path = 'data/embeddings_ada002_cache.pkl'

if os.path.exists(embeddings_ada002_cache_path):
    print("Loading cached OpenAI ada-002 embeddings...")
    with open(embeddings_ada002_cache_path, 'rb') as f:
        df_lyrics_ada = pickle.load(f)
    print(f"✓ Loaded {len(df_lyrics_ada)} embeddings from cache")
else:
    print("Generating OpenAI ada-002 embeddings...")
    df_lyrics_ada = pd.read_csv('data/lyrics_raw.csv')

    embeddings = []
    total = len(df_lyrics_ada)

    for idx, row in df_lyrics_ada.iterrows():
        if idx % 50 == 0:
            print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")

        emb = get_embedding_ada002(row['lyric_line'])
        embeddings.append(emb)
        time.sleep(0.05)

    df_lyrics_ada['embedding'] = embeddings

    # Validate
    dim = len(df_lyrics_ada['embedding'].iloc[0])
    assert dim == 1536, f"Wrong dimension! Expected 1536, got {dim}"

    # Cache
    with open(embeddings_ada002_cache_path, 'wb') as f:
        pickle.dump(df_lyrics_ada, f)

    print(f"✓ Generated {len(df_lyrics_ada)} embeddings (1536-dim)")

print(f"Embedding dimensions: {len(df_lyrics_ada['embedding'].iloc[0])}")
print(f"Total lines: {len(df_lyrics_ada)}")

# ========================================
# PHASE 2: Multi-Threshold Analysis
# ========================================
print("\n" + "=" * 60)
print("PHASE 2: Multi-Threshold Analysis")
print("=" * 60)

def multi_threshold_analysis(df, thresholds=[0.50, 0.55, 0.60, 0.65, 0.70]):
    """Calculate attention windows at multiple thresholds"""
    results = []

    for threshold in thresholds:
        print(f"Processing θ={threshold}...")

        for (album, song), group in df.groupby(['album', 'song']):
            windows = calculate_attention_window(group, threshold=threshold)

            for idx, window in enumerate(windows):
                results.append({
                    'threshold': threshold,
                    'album': album,
                    'artist': group.iloc[0]['artist'],
                    'song': song,
                    'attention_window': window
                })

    return pd.DataFrame(results)

print("Running multi-threshold analysis...")
df_multi = multi_threshold_analysis(df_lyrics_ada)

summary = df_multi.groupby(['threshold', 'artist'])['attention_window'].agg([
    'mean', 'median', 'std', 'count'
]).reset_index()

print("\n=== Multi-Threshold Summary ===")
print(summary)

# Save
df_multi.to_csv('data/multi_threshold_results.csv', index=False)
summary.to_csv('data/multi_threshold_summary.csv', index=False)
print("\n✓ Results saved")

# Visualization (fig9)
os.makedirs('2026-02-10-attention_windows', exist_ok=True)
plt.figure(figsize=(14, 6))

for artist, color in [('Pink Floyd', '#E91E63'), ('The Beatles', '#2196F3')]:
    data = summary[summary['artist'] == artist]
    plt.plot(data['threshold'], data['mean'], 'o-',
             color=color, linewidth=2.5, markersize=8, label=artist)
    plt.fill_between(data['threshold'],
                     data['mean'] - data['std'],
                     data['mean'] + data['std'],
                     alpha=0.2, color=color)

plt.xlabel('Threshold θ', fontsize=12)
plt.ylabel('Mean Attention Window (lines)', fontsize=12)
plt.title('Threshold Sensitivity Analysis: Crossover Detection', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('2026-02-10-attention_windows/fig9_threshold_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Figure 9 saved")

# Crossover analysis
print("\n=== Crossover Analysis ===")
for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
    floyd_mean = summary[(summary['artist'] == 'Pink Floyd') & (summary['threshold'] == threshold)]['mean'].values[0]
    beatles_mean = summary[(summary['artist'] == 'The Beatles') & (summary['threshold'] == threshold)]['mean'].values[0]
    ratio = floyd_mean / beatles_mean if beatles_mean > 0 else 0
    winner = 'Floyd' if floyd_mean > beatles_mean else 'Beatles'
    print(f"θ={threshold}: Floyd={floyd_mean:.2f}, Beatles={beatles_mean:.2f}, Ratio={ratio:.2f}, Winner={winner}")

# ========================================
# PHASE 3: Topic Modeling (BERTopic)
# ========================================
print("\n" + "=" * 60)
print("PHASE 3: Topic Modeling with BERTopic")
print("=" * 60)

try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    def topic_persistence_analysis(df):
        """Calculate topic persistence using BERTopic"""
        results = []

        for artist in ['Pink Floyd', 'The Beatles']:
            artist_df = df[df['artist'] == artist]

            for song, group in artist_df.groupby('song'):
                if len(group) < 5:
                    continue

                docs = group['lyric_line'].tolist()

                vectorizer = CountVectorizer(stop_words='english', min_df=1)
                model = BERTopic(vectorizer_model=vectorizer,
                               min_topic_size=2, nr_topics='auto', verbose=False)

                try:
                    topics, _ = model.fit_transform(docs)

                    topic_changes = sum([1 for i in range(len(topics)-1)
                                       if topics[i] != topics[i+1]])
                    persistence = len(topics) / (topic_changes + 1) if topic_changes > 0 else len(topics)

                    n_topics = len(set(topics)) - (1 if -1 in topics else 0)

                    results.append({
                        'artist': artist,
                        'song': song,
                        'topic_persistence': persistence,
                        'n_topics': n_topics,
                        'n_lines': len(topics)
                    })
                except Exception as e:
                    print(f"Skipping {song}: {str(e)[:50]}")

        return pd.DataFrame(results)

    print("Running BERTopic analysis...")
    df_topics = topic_persistence_analysis(df_lyrics_ada)

    print("\n=== Topic Modeling Results ===")
    topic_summary = df_topics.groupby('artist')[['topic_persistence', 'n_topics']].describe()
    print(topic_summary)

    df_topics.to_csv('data/topic_modeling_results.csv', index=False)
    print("\n✓ Results saved")

    # Visualization (fig10)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_topics, x='artist', y='topic_persistence',
                palette=['#E91E63', '#2196F3'])
    plt.title('Topic Persistence: Lines Before Topic Change',
              fontsize=14, fontweight='bold')
    plt.ylabel('Average Lines per Topic', fontsize=12)
    plt.xlabel('Artist', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('2026-02-10-attention_windows/fig10_topic_persistence.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Figure 10 saved")

except ImportError:
    print("⚠ BERTopic not installed. Skipping topic modeling.")
    print("Install with: pip install bertopic")
    df_topics = None

# ========================================
# PHASE 4: STS Analysis
# ========================================
print("\n" + "=" * 60)
print("PHASE 4: STS Analysis (Sentence Transformers)")
print("=" * 60)

try:
    from sentence_transformers import SentenceTransformer

    sts_cache_path = 'data/sts_embeddings_cache.pkl'

    if os.path.exists(sts_cache_path):
        print("Loading cached STS embeddings...")
        with open(sts_cache_path, 'rb') as f:
            df_lyrics_sts = pickle.load(f)
        print("✓ STS embeddings loaded from cache")
    else:
        print("Loading STS model (all-mpnet-base-v2)...")
        sts_model = SentenceTransformer('all-mpnet-base-v2')

        print("Generating STS embeddings...")
        df_lyrics_sts = df_lyrics_ada.copy()
        sts_embeddings = sts_model.encode(df_lyrics_sts['lyric_line'].tolist(),
                                           show_progress_bar=True)

        df_lyrics_sts['sts_embedding'] = list(sts_embeddings)

        with open(sts_cache_path, 'wb') as f:
            pickle.dump(df_lyrics_sts, f)

        print(f"✓ Generated {len(sts_embeddings)} STS embeddings (768-dim)")

    print(f"STS embedding dimensions: {len(df_lyrics_sts['sts_embedding'].iloc[0])}")

    # Calculate STS attention windows
    def calculate_sts_windows(df, threshold=0.70):
        """Calculate attention windows using STS embeddings"""
        results = []

        for (album, song), group in df.groupby(['album', 'song']):
            emb_matrix = np.array(group['sts_embedding'].tolist())

            for i in range(len(emb_matrix)):
                window_size = 0
                for j in range(i + 1, len(emb_matrix)):
                    sim = cosine_sim(emb_matrix[i], emb_matrix[j])
                    if sim > threshold:
                        window_size += 1
                    else:
                        break

                results.append({
                    'album': album,
                    'artist': group.iloc[0]['artist'],
                    'song': song,
                    'sts_window': window_size
                })

        return pd.DataFrame(results)

    print("Calculating STS-based attention windows...")
    df_sts = calculate_sts_windows(df_lyrics_sts)

    print("\n=== STS Attention Window Results ===")
    sts_summary = df_sts.groupby('artist')['sts_window'].describe()
    print(sts_summary)

    df_sts.to_csv('data/sts_attention_windows.csv', index=False)
    print("\n✓ Results saved")

    # Comparison visualization (fig11)
    df_windows_ada = []
    for (album, song), group in df_lyrics_ada.groupby(['album', 'song']):
        windows = calculate_attention_window(group, threshold=0.70)
        for idx, window in enumerate(windows):
            df_windows_ada.append({
                'album': album,
                'artist': group.iloc[0]['artist'],
                'song': song,
                'attention_window': window
            })
    df_windows_ada = pd.DataFrame(df_windows_ada)

    comparison = df_windows_ada.groupby('artist')['attention_window'].mean().reset_index()
    comparison = comparison.rename(columns={'attention_window': 'ada002_window'})
    sts_summary_mean = df_sts.groupby('artist')['sts_window'].mean().reset_index()
    comparison = comparison.merge(sts_summary_mean, on='artist')

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison))
    width = 0.35

    ax.bar(x - width/2, comparison['ada002_window'], width,
           label='OpenAI ada-002 (1536-dim)', color='#FF9800')
    ax.bar(x + width/2, comparison['sts_window'], width,
           label='STS MPNet (768-dim)', color='#4CAF50')

    ax.set_ylabel('Mean Attention Window (lines)', fontsize=12)
    ax.set_title('Comparison: OpenAI ada-002 vs STS-based Attention Windows',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['artist'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('2026-02-10-attention_windows/fig11_sts_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Figure 11 saved")

except ImportError:
    print("⚠ sentence-transformers not installed. Skipping STS analysis.")
    print("Install with: pip install sentence-transformers")
    df_sts = None
    df_lyrics_sts = None

# ========================================
# PHASE 5: Validation
# ========================================
print("\n" + "=" * 60)
print("PHASE 5: Validation Checks")
print("=" * 60)

# 1. Embedding quality
print("\n1. Adjacent Line Similarity Check")
print("-" * 50)
adj_sims = []
for i in range(len(df_lyrics_ada)-1):
    if df_lyrics_ada.loc[i, 'song'] == df_lyrics_ada.loc[i+1, 'song']:
        sim = cosine_sim(df_lyrics_ada.loc[i, 'embedding'],
                        df_lyrics_ada.loc[i+1, 'embedding'])
        adj_sims.append(sim)

adj_sim_mean = np.mean(adj_sims)
print(f"Adjacent line similarity: {adj_sim_mean:.3f}")
print(f"Expected range: 0.25-0.35")
print(f"Status: {'✓ PASS' if 0.20 < adj_sim_mean < 0.40 else '⚠ WARNING'}")

# 2. Bootstrap validation
print("\n2. Bootstrap Confidence Interval Validation")
print("-" * 50)
floyd_windows_ada = df_windows_ada[df_windows_ada['artist'] == 'Pink Floyd']['attention_window']
beatles_windows_ada = df_windows_ada[df_windows_ada['artist'] == 'The Beatles']['attention_window']

diffs = []
for _ in range(2000):
    f_sample = np.random.choice(floyd_windows_ada, size=len(floyd_windows_ada), replace=True)
    b_sample = np.random.choice(beatles_windows_ada, size=len(beatles_windows_ada), replace=True)
    diffs.append(np.mean(f_sample) - np.mean(b_sample))

ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
print(f"95% CI (Floyd - Beatles): [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Contains zero: {ci_low < 0 < ci_high}")
print(f"Status: {'✓ PASS - Significant difference' if not (ci_low < 0 < ci_high) else '⚠ WARNING'}")

# 3. Consistency matrix
print("\n3. Consistency Across Metrics")
print("-" * 50)

consistency_data = {
    'Metric': [
        'Attention Windows (θ=0.70)',
        'Attention Windows (θ=0.60)',
        'Attention Windows (θ=0.50)',
        'STS Windows',
        'Topic Persistence'
    ],
    'Floyd_Mean': [0, 0, 0, 0, 0],
    'Beatles_Mean': [0, 0, 0, 0, 0],
    'Winner': ['', '', '', '', '']
}

# Fill data
floyd_070 = df_windows_ada[df_windows_ada['artist'] == 'Pink Floyd']['attention_window'].mean()
beatles_070 = df_windows_ada[df_windows_ada['artist'] == 'The Beatles']['attention_window'].mean()
consistency_data['Floyd_Mean'][0] = round(floyd_070, 2)
consistency_data['Beatles_Mean'][0] = round(beatles_070, 2)
consistency_data['Winner'][0] = 'Floyd' if floyd_070 > beatles_070 else 'Beatles'

floyd_060 = summary[(summary['artist'] == 'Pink Floyd') & (summary['threshold'] == 0.60)]['mean'].values[0]
beatles_060 = summary[(summary['artist'] == 'The Beatles') & (summary['threshold'] == 0.60)]['mean'].values[0]
consistency_data['Floyd_Mean'][1] = round(floyd_060, 2)
consistency_data['Beatles_Mean'][1] = round(beatles_060, 2)
consistency_data['Winner'][1] = 'Floyd' if floyd_060 > beatles_060 else 'Beatles'

floyd_050 = summary[(summary['artist'] == 'Pink Floyd') & (summary['threshold'] == 0.50)]['mean'].values[0]
beatles_050 = summary[(summary['artist'] == 'The Beatles') & (summary['threshold'] == 0.50)]['mean'].values[0]
consistency_data['Floyd_Mean'][2] = round(floyd_050, 2)
consistency_data['Beatles_Mean'][2] = round(beatles_050, 2)
consistency_data['Winner'][2] = 'Floyd' if floyd_050 > beatles_050 else 'Beatles'

if df_sts is not None:
    floyd_sts = df_sts[df_sts['artist'] == 'Pink Floyd']['sts_window'].mean()
    beatles_sts = df_sts[df_sts['artist'] == 'The Beatles']['sts_window'].mean()
    consistency_data['Floyd_Mean'][3] = round(floyd_sts, 2)
    consistency_data['Beatles_Mean'][3] = round(beatles_sts, 2)
    consistency_data['Winner'][3] = 'Floyd' if floyd_sts > beatles_sts else 'Beatles'

if df_topics is not None:
    floyd_topics = df_topics[df_topics['artist'] == 'Pink Floyd']['topic_persistence'].mean()
    beatles_topics = df_topics[df_topics['artist'] == 'The Beatles']['topic_persistence'].mean()
    consistency_data['Floyd_Mean'][4] = round(floyd_topics, 2)
    consistency_data['Beatles_Mean'][4] = round(beatles_topics, 2)
    consistency_data['Winner'][4] = 'Floyd' if floyd_topics > beatles_topics else 'Beatles'

df_consistency = pd.DataFrame(consistency_data)
print(df_consistency)

df_consistency.to_csv('data/consistency_matrix.csv', index=False)
print("\n✓ Consistency matrix saved")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print("\nGenerated files:")
print("  ✓ data/embeddings_ada002_cache.pkl")
print("  ✓ data/multi_threshold_results.csv")
print("  ✓ data/multi_threshold_summary.csv")
if df_topics is not None:
    print("  ✓ data/topic_modeling_results.csv")
if df_sts is not None:
    print("  ✓ data/sts_attention_windows.csv")
    print("  ✓ data/sts_embeddings_cache.pkl")
print("  ✓ data/consistency_matrix.csv")
print("  ✓ fig9_threshold_sensitivity.png")
if df_topics is not None:
    print("  ✓ fig10_topic_persistence.png")
if df_sts is not None:
    print("  ✓ fig11_sts_comparison.png")
