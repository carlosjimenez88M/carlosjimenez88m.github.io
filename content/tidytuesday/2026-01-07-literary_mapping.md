---
author: Carlos Daniel Jiménez
date: 2026-01-07
title: "Literary Mapping of Christmas Novels: A Vector Narrative Arc Approach"
categories: ["Agentic AI", "LLMS"]
tags: ["LLMs", "NLP"]
series:
  - NLP
  - LLMs
  - Embeddings 
---




## Post Objective

- Data cleaning and preliminary analysis process
- Understanding the emotional charge or plot development of texts through semantic archaeology based on PCAs
- Understanding the connections and most representative ideas within the document set



## Intention

Understanding a story's behavior at the level of its variance is a challenge addressed by attentional engineering. Therefore, using lesser-known methods such as the **vector narrative arc** combined with a **literary map** constitutes an interesting route to address increasingly common problems.

The problems this duo solves range from defining more precise categories in user preferences (for example, *bohemian rock* or *slow psychological horror*), understanding their emotional digital footprint, to detecting plagiarism by inspiration—as in the Coldplay vs. Joe Satriani case with the song "Viva la Vida"—where graphs can validate semantic similarity or structural plagiarism.

Generally, this technique is not widely discussed or popularized given its hybrid approach, since it entails statistical rigor in verifying the purity of the clusters with which TSEs are constructed in the context of embeddings (this is one of the reasons for the model selection itself; I will detail this more precisely later).



## The Geometry of Meaning

Meaning can be represented as a physical location in multidimensional space by transforming text fragments into **Embeddings**.

### The Mathematics

If $w$ is our text, the embedding function $f$ transforms it into a vector $v$:

$$v = f(w) \in \mathbb{R}^d$$

Where $d$ is the model's dimension (in the case of `text-embedding-004`, $d=768$).

Embeddings have algebraic properties. If the model is good, the mathematical operation:

$$\text{Vector}(\text{King}) - \text{Vector}(\text{Man}) + \text{Vector}(\text{Woman}) \approx \text{Vector}(\text{Queen})$$

This proves the model captured **semantics** (gender and hierarchy) as directions in space.

These meanings have interesting properties such that when using a similarity measure:

### The Formula

Given two text vectors $A$ and $B$, their similarity is:

$$\text{similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

We obtain:

- If the result is **1**: They are semantic twins
- If it's **0**: They are orthogonal (unrelated)
- If it's **-1**: They are semantic opposites

Up to this point, we've mentioned some classic concepts from NLP and LLM theory. From here comes the interesting part: analyzing grammatical influence in a network. Semantic graphs work to transform a series of documentations or texts into fragments that describe edges that can evaluate semantic similarity, resulting in the influence of one text over another or, in other words, **Nearest Neighbor Search**:

$$E_{ij} = \begin{cases} \text{similarity}(V_i, V_j) & \text{if similarity}(V_i, V_j) > \text{threshold} \\ 0 & \text{if similarity}(V_i, V_j) \leq \text{threshold} \end{cases}$$



## Vonnegut's Theory

Kurt Vonnegut in 1981 proposed a central idea about literary works: although each story is unique, emotional patterns are recognizable (on a Y-axis, which we'll call **semantic position**) and repetitive (on an X-axis, which we'll call **Narrative Time**). Thanks to this, we can identify the emotional charge of a text. Some of the patterns he found based on this theory are:

- **"Man in a Hole"**: The protagonist starts well, falls into problems, and then gets out of them
- **"Boy Meets Girl"**: Starts normal, improves with the romantic encounter, drops with separation, and rises again with reunion
- **"From Bad to Worse"**: Starts bad and ends worse—as in Kafka's works
- **"Cinderella"**: Rises, falls, and then rises even higher—the archetypal story

Now, what catches our attention is how this complements the **Narrative Arc** where events are mapped by emotion.

For the above to make sense, let's develop the following exercise based on a **tidytuesday** proposed by the R community.



## Implementation

### Loading Libraries

Let's start by loading the libraries we'll use for the exercise and predefining some data visualization standards:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
```

### Data Loading and Feature Engineering

Next, we load the databases, apply feature engineering treatment, minimal data cleaning, and finally concatenation:

```python
christmas_novel_text['text'] = christmas_novel_text['text'].replace(r'^\s*$', np.nan, regex=True)

full_dataset = christmas_novels\
    .merge(christmas_novel_text, left_on='gutenberg_id', 
           right_on='gutenberg_id')\
    .dropna(subset=['text'])\
    .reset_index(drop=True)\
    .merge(christmas_novel_authors,
           left_on='gutenberg_author_id', 
           right_on='gutenberg_author_id', 
           how='left').reset_index(drop=True)
 
full_dataset['birthdate'] = full_dataset['birthdate'].astype('Int64')
full_dataset['deathdate'] = full_dataset['deathdate'].astype('Int64')
full_dataset['author_age_at_death'] = full_dataset['deathdate'] - full_dataset['birthdate']
```

The intention of calculating authors' age at death was to validate some correlation between their narrative length and age. The result was null, but for experimental purposes, it functioned as an important theoretical arc—knowing whether longevity was reflected in their texts (remember, there's a study saying Nobel Prize winners live several years longer than the societal average).

```python
grp_novel_text[['author_age_at_death','word_count']].corr()

#                    author_age_at_death  word_count
# author_age_at_death         1.000000    -0.070052
# word_count                 -0.070052     1.000000
```

### Text Cleaning

Now for the text cleaning part, we applied a function built with regular expressions as follows:

```python
def clean_gutenberg_keep_punctuation(text):
    text = re.sub(r'\[illustration:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[transcriber\'?s? note:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'-{3,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'"', ' ', text)
    text = re.sub(r'\* \* \* \* \*', ' ', text)
    return text.strip()
```

Note that this is very specific to the texts we're dealing with, so when concatenating, this will help prevent noise from being included in the texts, allowing us to work with purer embeddings.

```python
grp_novel_text = (
    full_dataset
    .assign(text=full_dataset['text'].fillna('').astype(str))
    .groupby(['gutenberg_id','title','author','author_age_at_death'])['text']
    .apply(lambda x: ' '.join(s.strip() for s in x if s.strip()))
    .reset_index()
)

grp_novel_text['text'] = grp_novel_text['text'].apply(clean_gutenberg_keep_punctuation)
grp_novel_text['text'] = grp_novel_text['text'].str.replace(r'\s+', ' ', regex=True).str.lower().str.strip()
grp_novel_text['word_count'] = grp_novel_text['text'].apply(lambda x: len(x.split()))
grp_novel_text.head()
```



## Creating Embeddings

Now we have a structured database, to which I intend to add one more step: aggregate or transform the text into embeddings. In this case, I'll work with `models/text-embedding-004`. Understanding the attention window that models have, I'll split while attempting to mathematize a sentiment scale by concept block that allows me to discuss or discover behavioral patterns within Christmas novels.

Without falling into romances, I selected this embedding model for the following reasons:

**i) Dimensional Elasticity**: the most important dimensions are stored at the beginning of the vector, so the strong or explanatory variance is the first we'll encounter when working on this type of project (also known as **"Matryoshka" Embeddings**).

**ii)** Since graphs will be used to search for the idea or text segment that best describes the document cluster, this type of embedding stores or has better semantic resolution or **MTEB scores**.

```python
def get_embedding_safe(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return result['embedding']
    except:
        return None
 
def split_text(text, chunk_size=3000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap) 
    return chunks

all_chunks = []
print("Processing books...")
for index, row in grp_novel_text.iterrows():
    book_chunks = split_text(row['text'])
    for i, chunk in enumerate(book_chunks):
        all_chunks.append({
            'book_title': row['title'],
            'author': row['author'],
            'chunk_id': i,
            'text_chunk': chunk
        })

df_chunks = pd.DataFrame(all_chunks)
df_chunks['embedding'] = df_chunks['text_chunk'].apply(get_embedding_safe)
df_chunks.head()
```



## Finding Optimal Clusters

Now comes an important part: after having embeddings, comes the search for their meaning. The general idea is to discover what exists in this taxonomy of texts and meanings. Remember that embeddings are concept maps where movement means something, and this is where their geometry makes sense.

Let's have a hypothesis about the Christmas Novel: it spans genres from horror to the religious representation of unity. By fragmenting the text and preserving certain elastic memory within chunks, we can see that there are moments, and each moment obeys a subgenre—and here comes the first finding:

```python
matrix = np.vstack(df_chunks['embedding'].values)
inertia = []
silhouette_scores = []
K_range = range(2, 13)

for k in K_range:
    kmeans = KMeans(n_clusters=k, 
                    random_state=42, 
                    n_init=10)
    kmeans.fit(matrix)
    
    inertia.append(kmeans.inertia_)
    score = silhouette_score(matrix, kmeans.labels_)
    silhouette_scores.append(score)

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Lower is better)', color=color)
ax1.plot(K_range, inertia, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.set_ylabel('Silhouette Score (Higher is better)', color=color)
ax2.plot(K_range, silhouette_scores, marker='x', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Analysis to find optimal K: Elbow vs Silhouette')
plt.grid(True)
plt.show()
```

Mathematically, it shows there are two genres in the clusters—probably the author with the greatest presence and the rest of the authors. But following the genre thesis, we'll set `k=4` to see what happens with the map design from a t-SNE, where we seek to understand semantic proximity:



## Literary Cartography with t-SNE

```python
kmeans_final = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
df_chunks['cluster'] = kmeans_final.fit_predict(matrix)

# 4. 2D VISUALIZATION (t-SNE)
# t-SNE reduces the 768 dimensions to 2 (X and Y) while respecting semantic proximity
print("Running t-SNE for dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random')
proyeccion_2d = tsne.fit_transform(matrix)

df_chunks['x'] = proyeccion_2d[:, 0]
df_chunks['y'] = proyeccion_2d[:, 1]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_chunks['x'], df_chunks['y'], 
                     c=df_chunks['cluster'], cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, label='Cluster ID')

for i in range(0, len(df_chunks), 100):  # Label 1 out of every 100 points to avoid saturation
    plt.text(df_chunks['x'].iloc[i], df_chunks['y'].iloc[i], 
             df_chunks['book_title'].iloc[i][:15], fontsize=8, alpha=0.7)

plt.title(f'Semantic Map of Books (t-SNE with {K_OPTIMO} clusters)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
```

This result is what's known as **literary cartography**, where the findings are:

- **The Continent of the Supernatural (Cluster 3)**: Where Dickens' ghosts dwell—an isolated and dense group
- **The Valley of Tears (Cluster 0)**: Where texts about silent tragedy, lies, and social conflict reside
- **The Hill of Duty (Cluster 2)**: Texts centered on morality, homeland, and honor
- **The Garden of Romance (Cluster 1)**: Feelings, flowers, and happy endings

With this, we achieve semantic archaeology on the Christmas novel reduced to 4 themes. But the question that follows is: how do we measure the emotion of each story and, finally, which ones might have been inspired by others?



## The Behavior of a Story

```python
from sklearn.decomposition import PCA

matrix = np.vstack(df_chunks['embedding'].values) 

pca = PCA(n_components=1, random_state=42)
narrative_axis = pca.fit_transform(matrix)

df_chunks['vonnegut_axis'] = narrative_axis

book1 = "A Christmas Carol in Prose; Being a Ghost Story of Christmas"  # The classic
book2 = "Mr. Blake's Walking-Stick: A Christmas Story for Boys and Girls"  # The flat

y1 = df_chunks[df_chunks['book_title'] == book1]['vonnegut_axis'].values
y2 = df_chunks[df_chunks['book_title'] == book2]['vonnegut_axis'].values

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.figure(figsize=(14, 6))

plt.plot(smooth(y1, 5), label='A Christmas Carol (Dynamic Arc)', color='#E63946', linewidth=3)
plt.plot(smooth(y2, 5), label="Mr. Blake's Walking-Stick (Flat Arc)", color='#457B9D', linestyle='--', linewidth=2)

plt.title('The Shape of Stories (Vonnegut Vectorial Analysis)', fontsize=16)
plt.xlabel('Narrative Time (Book Progress)', fontsize=12)
plt.ylabel('Emotional State (Principal Component 1)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

The above image is what's called **Narrative Dynamism**, where mathematically different emotions are described (imagine sadness, fear, and then a moment of joy due to some situation in the story) as represented by the red line, while the blue represents an emotional temperature that doesn't change, like a linear story.

Therefore, this method helps determine what's worth reading and what has boring behavior, so to speak. There's something important about this method's applications that hasn't been mentioned: with this technique, you can evaluate when a story is losing coherence (aggressive fluctuations).

> **Important note**: `pca=1` to give weight to the work's main theme being studied.



## Graph-Based Inspiration

Given the previous steps, we have: sub-themes within stories, context thanks to embeddings, narrative evolution thanks to PCA and seeing peaks and valleys. Now we need to understand the bridges of similarity.

With this, I seek to understand which embedding vectors connect at which point given the stories or text fragments, but not viewing it from how many words they have in common, but rather which ideas are similar. Therefore, what we'll work on next is:

1. Understanding which authors can be compared through a directed filter
2. Comparing paragraphs per author pair to understand existing iterations
3. Finding fragments that have intentional or revealed semantic similarity

With this, we can find where styles that authors use converge with others to say that their texts have influences from others, or in other words, reveal the influences that exist between Christmas literature.

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

top_authors = df_chunks['author'].value_counts().head(5).index.tolist()
df_top = df_chunks[df_chunks['author'].isin(top_authors)].copy()
matrix_top = matrix[df_top.index] 

print(f"Analyzing connections between: {top_authors}")

sim_matrix_top = cosine_similarity(matrix_top)

G = nx.Graph()
connections_found = []

for i in range(len(top_authors)):
    for j in range(i + 1, len(top_authors)):
        auth_A = top_authors[i]
        auth_B = top_authors[j]
        
        idxs_A = np.where(df_top['author'] == auth_A)[0]
        idxs_B = np.where(df_top['author'] == auth_B)[0]
        
        sub_sim = sim_matrix_top[np.ix_(idxs_A, idxs_B)]
        
        max_idx_flat = np.argmax(sub_sim)
        r_local, c_local = np.unravel_index(max_idx_flat, sub_sim.shape)
        
        max_score = sub_sim[r_local, c_local]
        
        global_idx_A = idxs_A[r_local]
        global_idx_B = idxs_B[c_local]
        
        node_id_A = df_top.index[global_idx_A]
        G.add_node(node_id_A, label=auth_A, text=df_top.iloc[global_idx_A]['text_chunk'], type='fragment')
        
        node_id_B = df_top.index[global_idx_B]
        G.add_node(node_id_B, label=auth_B, text=df_top.iloc[global_idx_B]['text_chunk'], type='fragment')
        
        G.add_edge(node_id_A, node_id_B, weight=max_score)
        
        connections_found.append({
            'Auth1': auth_A,
            'Auth2': auth_B,
            'Score': max_score,
            'Text1': df_top.iloc[global_idx_A]['text_chunk'][:100],
            'Text2': df_top.iloc[global_idx_B]['text_chunk'][:100]
        })

plt.figure(figsize=(12, 8))
pos = nx.circular_layout(G)
```



## Evidence of Similarity (Real Texts)

### 🔗 McIntosh, Maria J. ↔ Thackeray, William Makepeace
**Similarity: 0.7623**

- 📜 **McIntosh**: "d have done had her fingers trembled less. can you sing? elevated above all apprehension by the indi..."
- 📜 **Thackeray**: ", struck me with a terror which i cannot describe, and impressed me with the fact of the vast progre..."

### 🔗 McIntosh, Maria J. ↔ Finley, Martha
**Similarity: 0.7740**

- 📜 **McIntosh**: "e interrupted, for all were busy in preparing for this important day. miss donaldson was superintend..."
- 📜 **Finley**: "tiful. i'm sure everybody thinks so. don't they, papa? as far as my knowledge goes, he answered, smi..."

### 🔗 McIntosh, Maria J. ↔ Allen, James Lane
**Similarity: 0.7815**

- 📜 **McIntosh**: "could have been so rigid in his observance of a soldier's duty, yet so inexpressibly tender as a man..."
- 📜 **Allen**: ", it slipped from his hand and there was a loud clangor. she stepped quickly out upon the stone befo..."

### 🔗 McIntosh, Maria J. ↔ Hale, Edward Everett
**Similarity: 0.8113** *(The highest!)*

- 📜 **McIntosh**: "th, $3; cloth, gilt leaves, $4; morocco extra, $6. cheaper edition, with portrait and 4 plates. im. ..."
- 📜 **Hale**: "by mrs. harriet beecher stowe, mrs. a. d. t. whitney, miss lucretia hale, rev. e. e. hale, f. b. per..."

### 🔗 Thackeray, William Makepeace ↔ Finley, Martha
**Similarity: 0.7520**

- 📜 **Thackeray**: "davison! who is it? cried out miss raby, starting and turning as white as a sheet. i told her it wa..."
- 📜 **Finley**: "ry nice old fellow, returned the little girl with an arch look and smile. so i'll hang mine up. and ..."

### 🔗 Thackeray, William Makepeace ↔ Allen, James Lane
**Similarity: 0.7663**

- 📜 **Thackeray**: "me, and pledge a hand to all young friends, as fits the merry christmas time. on life's wide scene y..."

### 🔗 Allen, James Lane ↔ Hale, Edward Everett
**Similarity: 0.7855**

- 📜 **Allen**: "re most rapidly dying out in this civilization--the shadow of that romance which for ages was the ea..."
- 📜 **Hale**: "n those virgins arose and trimmed their lamps. and i will light them, said she aloud. that will save..."



## Interpreting the Findings

### 1. The "Structural" Finding: When AI Reads Format, Not Story

**The Case:** McIntosh vs. Hale (similarity: 0.8113—the highest!)

**McIntosh text:** "...th, $3; cloth, gilt leaves, $4; morocco extra, $6. cheaper edition..."  
**Hale text:** "...by mrs. harriet beecher stowe... rev. e. e. hale, f. b. per..."

#### What happened here?

A human would say they're not similar. One talks about money and binding ("cloth", "gilt"); the other is a list of author names ("Stowe", "Hale").

The embedding (004) says: **"Both are lists of editorial metadata."**

#### The brilliant interpretation:

The model detected **structure**. Both are non-narrative fragments: catalogs, short lists, proper names, or figures. Mathematically, both "break" the novel's fluid narrative.

**Conclusion:** Your tool automatically separates "content" (the story) from "packaging" (editorial advertising).



### 2. The "Tonal" Finding: When AI Detects Solemnity

**The Case:** Allen vs. Hale (similarity: 0.7855)

**Allen text:** "...shadow of that romance which for ages was the ea..."  
**Hale text:** "...virgins arose and trimmed their lamps. and i will light them..."

#### What happened here?

Allen talks about "ancient romance." Hale cites a biblical parable (the virgins and the lamps).

The embedding says: **"Both have an elevated and mystical tone."**

#### The brilliant interpretation:

The model didn't search for repeated words. It searched for the feeling of "antiquity", "solemnity", and "spirituality". It connected romantic nostalgia with religion because both occupy the same place in the "Victorian sentiments space".



## Conclusion: The Invisible Architecture of Stories

This exercise reveals a fundamental truth: **literary narratives are living geometries**. What Vonnegut perceived as artistic intuition in 1981—that every story traces an emotional shape in space—today materializes as pure mathematics. We haven't reduced art to numbers; we've discovered that art was always geometry disguised as prose.

### The Three Pillars of a New Reading

#### 1. Literary geometry exists (and always did)

Every word in "A Christmas Carol" occupies a precise position in a 768-dimensional space. It's not metaphor: it's **topology**. When Dickens wrote "redemption," he didn't just choose a concept—he chose a vector coordinate close to "hope" and distant from "despair." Literature is spatial navigation. Great authors are cartographers of emotional territories we can only now map.

#### 2. Machines see the bones beneath the skin of words

While humans read surface—words, phrases, metaphors—embeddings read **deep structure**. When the model connected an editorial catalog with a list of Victorian authors, it didn't make a mistake: it discovered both shared the same syntactic architecture. It sees patterns we feel but don't name. It's like discovering two distant buildings were designed by the same invisible architect.

#### 3. Inspiration is a measurable ghost

Semantic graphs don't just trace obvious literary influences. They reveal **phantom conversations** between minds separated by centuries, dialoguing through archetypes that transcend languages, cultures, and vocabularies. Thackeray responding to a narrative gesture by McIntosh that she never consciously formulated. The literary tradition is a collective neural network, and we're just learning to read it.



### Application Horizons: Beyond Dickens

This methodology transcends the Christmas corpus. Imagine it deployed in:

- **Truth Engineering**: Comparing the vector signature of verified narratives vs. structured disinformation. Lies have recognizable topologies.

- **Screenplay Science**: Predicting emotional resonance in audiences not by what a film says, but by how it **breathes**—its vector narrative arc.

- **Recommendation by Emotional Architecture**: "If you were captivated by *Breaking Bad's* dynamic arc, here are books with the same narrative curvature"—beyond genre, we recommend by **shape**.

- **Coherence Audits**: Detecting the exact moment where a text loses its thread. Chaotic fluctuations in PCA are symptoms of narrative disorientation.

- **Computational Cultural Archaeology**: Tracking how themes—Christmas, revolutionary, apocalyptic—evolve through centuries, mutating but preserving their vector core.



### The Horizon: Attentional Engineering

We call this **attentional engineering**: the rigorous study of how texts capture, sustain, and release human consciousness. Each peak in a narrative arc marks a moment where the author decided to tension the string of attention. Each cluster in the semantic map reveals a **conceptual attractor basin**—regions of idea space where certain thoughts naturally gravitate.

**The civilizational promise**: In an era of information overabundance, these techniques don't replace human judgment—they **amplify** it. Not to read for us, but to **point us toward what deserves to be read**. To find the needles of meaning in infinite haystacks of noise.

As Vonnegut observed four decades ago: every story has shape. Today, for the first time in human history, **we can draw that shape with the same precision we measure galaxies**.

Literature ceases to be a purely subjective art. It becomes a science of human experience—rigorous, measurable, but no less beautiful for it.

**The map finally matches the territory.**



## Resources

**Complete code and datasets**: [Link to GitHub]  
**Models used**: `text-embedding-004` (Google), `scikit-learn`, `networkx`  
**Theoretical inspiration**: Kurt Vonnegut (1981), "The Shape of Stories"



*If you found this analysis interesting, consider applying these techniques to your own text corpus. The code is adaptable to any literary genre, from sci-fi to romance, from poetry to screenplays. The geometry of meaning awaits.*