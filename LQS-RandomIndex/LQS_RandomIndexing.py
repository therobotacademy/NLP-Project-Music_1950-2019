# %%
import pandas as pd
import numpy as np
import re
from numpy.linalg import norm
from tqdm import tqdm

# ---------------- Random Indexing Embeddings (fallback to Word2Vec) ----------------
def tokenize_lines(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    lines = [ln.strip() for ln in re.split(r'\r?\n+', text) if ln.strip()]
    # simple tokenization: lowercase, keep alphabetic+digits+accented
    tokens_per_line = []
    for ln in lines:
        toks = re.findall(r"[A-Za-zÀ-ÿ0-9']+", ln.lower())
        toks = [t for t in toks if len(t) > 1]
        if toks:
            tokens_per_line.append(toks)
    return tokens_per_line

def build_vocab(sentences, min_count=2, max_vocab=30000):
    from collections import Counter
    cnt = Counter()
    for s in sentences:
        cnt.update(s)
    # filter by min_count and cap vocab size
    items = [(w, c) for w, c in cnt.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_vocab]
    itos = [w for w, c in items]
    stoi = {w:i for i,w in enumerate(itos)}
    freqs = np.array([c for w,c in items], dtype=np.int64)
    return stoi, itos, freqs

def make_random_index_vectors(vocab_size, dim=200, nonzeros=4, seed=123):
    rng = np.random.default_rng(seed)
    R = np.zeros((vocab_size, dim), dtype=np.float32)
    for i in range(vocab_size):
        idxs = rng.choice(dim, size=nonzeros, replace=False)
        signs = rng.choice([-1.0, 1.0], size=nonzeros)
        R[i, idxs] = signs
    return R

def train_random_indexing(sentences, stoi, dim=200, window=4, nonzeros=4, seed=123):
    V = len(stoi)
    R = make_random_index_vectors(V, dim=dim, nonzeros=nonzeros, seed=seed)
    S = np.zeros((V, dim), dtype=np.float32)  # semantic vectors
    for s in sentences:
        ids = [stoi[w] for w in s if w in stoi]
        n = len(ids)
        for i, wid in enumerate(ids):
            left = max(0, i-window); right = min(n, i+window+1)
            context_ids = ids[left:i] + ids[i+1:right]
            if not context_ids:
                continue
            # sum of random index vectors of context
            ctx_vec = R[context_ids].sum(axis=0)
            S[wid] += ctx_vec
    # normalize
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    S = S / norms
    return S, R  # semantic embeddings, index vectors

def word_vec(word, stoi, S):
    i = stoi.get(word, None)
    if i is None: 
        return None
    return S[i]

def average_vec(tokens, stoi, S):
    vecs = [word_vec(t, stoi, S) for t in tokens if t in stoi]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def cos(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0: 
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def coherence_prime(verse_vecs, mu=0.65, sigma=0.15):
    if len(verse_vecs) < 2:
        return 0.0
    sims = [cos(verse_vecs[i-1], verse_vecs[i]) for i in range(1, len(verse_vecs))]
    C = float(np.mean(sims)) if sims else 0.0
    Cp = float(np.exp(-((C - mu)**2) / (2 * sigma**2)))
    return max(0.0, min(1.0, Cp))

def diversity_score(verse_vecs):
    n = len(verse_vecs)
    if n < 2:
        return 0.0
    # Matriz normalizada para cosenos vectorizados
    V = np.vstack(verse_vecs)
    # Evita divisiones por cero
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Vn = V / norms
    # Similaridad promedio pares i<j
    S = Vn @ Vn.T
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    if sims.size == 0:
        return 0.0
    raw = 1.0 - float(np.mean(sims))
    # Normalización heurística a [0,1] asumiendo cos ~ [0.3, 0.9] -> raw in [0.1, 0.7]
    return float(np.clip((raw - 0.1) / (0.7 - 0.1), 0, 1))

def novelty_score(verse_vecs, genre_centroid):
    if genre_centroid is None or len(verse_vecs) == 0:
        return 0.0
    sims = [cos(v, genre_centroid) for v in verse_vecs]
    raw = 1.0 - float(np.mean(sims))
    return float(np.clip((raw - 0.1) / (0.7 - 0.1), 0, 1))

def lqs_score(Cp, D, N, w=(0.4, 0.35, 0.25)):
    return float(w[0]*Cp + w[1]*D + w[2]*N)

# ---------------- Load data ----------------
file_path = "../data/tcc_ceds_music.csv"
df = pd.read_csv(file_path)
df = df[['artist_name', 'track_name', 'genre', 'lyrics']].dropna(subset=['lyrics', 'genre', 'artist_name'])
df = df[df['lyrics'].str.len() > 0]

# Subset
# random_state=123 for reproducibility
N_SAMPLE = 500
df_sample = df.sample(N_SAMPLE, random_state=123).reset_index(drop=True)

# ---------------- Tokenize ----------------
song_verses = []
for txt in tqdm(df_sample['lyrics'].tolist(), desc="Tokenizing"):
    tokens_by_line = tokenize_lines(txt)
    song_verses.append(tokens_by_line)

# Construye corpus de oraciones (versos)
sentences = [line for song in song_verses for line in song]

if len(sentences) < 10:
    raise RuntimeError("Muy pocas oraciones para entrenar embeddings en el subconjunto.")

# ---------------- Train Random Indexing ----------------
stoi, itos, freqs = build_vocab(sentences, min_count=2, max_vocab=30000)
S, R = train_random_indexing(sentences, stoi, dim=200, window=4, nonzeros=4, seed=123)

# ---------------- Verse embeddings per song ----------------
song_verse_vecs = []
song_mean_vecs = []
for tokens_by_line in song_verses:
    vvecs = []
    for toks in tokens_by_line:
        v = average_vec(toks, stoi, S)
        if v is not None:
            vvecs.append(v)
    song_verse_vecs.append(vvecs)
    if len(vvecs) > 0:
        song_mean_vecs.append(np.mean(np.vstack(vvecs), axis=0))
    else:
        song_mean_vecs.append(np.zeros(S.shape[1], dtype=np.float32))

df_sample['song_vec'] = song_mean_vecs

# ---------------- Genre centroids ----------------
genre_means = {}
for g, grp in df_sample.groupby('genre'):
    mats = np.vstack(grp['song_vec'].values) if len(grp)>0 else None
    if mats is not None and mats.size > 0:
        genre_means[g] = np.mean(mats, axis=0)
    else:
        genre_means[g] = None

# ---------------- Compute metrics & LQS ----------------
rows = []
for i, row in df_sample.iterrows():
    verses_vecs = song_verse_vecs[i]
    Cp = coherence_prime(verses_vecs)
    D = diversity_score(verses_vecs)
    gcent = genre_means.get(row['genre'])
    N = novelty_score(verses_vecs, gcent)
    LQS = lqs_score(Cp, D, N, w=(0.4, 0.35, 0.25))
    rows.append((row['artist_name'], row['genre'], row['track_name'], Cp, D, N, LQS))

res_df = pd.DataFrame(rows, columns=['artist_name', 'genre', 'track_name', 'coherence_prime', 'diversity', 'novelty', 'LQS'])

# ---------------- Aggregations ----------------
by_genre = res_df.groupby('genre').agg(
    LQS_mean=('LQS', 'mean'),
    LQS_std=('LQS', 'std'),
    n=('LQS', 'count')
).reset_index().sort_values('LQS_mean', ascending=False)

by_artist_genre = res_df.groupby(['artist_name', 'genre']).agg(
    LQS_mean=('LQS', 'mean'),
    n=('LQS', 'count')
).reset_index().sort_values(['LQS_mean', 'n'], ascending=[False, False])

# Save outputs
subset_path = "./results/lqs_results_subset.csv"
genre_path = "./results/lqs_genre_summary.csv"
artist_genre_path = "./results/lqs_artist_genre_summary.csv"
res_df.to_csv(subset_path, index=False)
by_genre.to_csv(genre_path, index=False)
by_artist_genre.to_csv(artist_genre_path, index=False)