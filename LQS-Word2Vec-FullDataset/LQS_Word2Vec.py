# %%
import pandas as pd
import numpy as np
import re
from numpy.linalg import norm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm

# ---------- Utils ----------
def tokenize_lines(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # Divide en líneas/versos y tokeniza cada línea (minúsculas, alfanumérico)
    lines = [ln.strip() for ln in re.split(r'\r?\n+', text) if ln.strip()]
    tokens_per_line = [simple_preprocess(ln, deacc=False, min_len=2) for ln in lines]
    # Filtra líneas vacías
    tokens_per_line = [toks for toks in tokens_per_line if len(toks) > 0]
    return tokens_per_line

def average_vec(tokens, model):
    if not tokens:
        return None
    vecs = []
    for t in tokens:
        if t in model.wv:
            vecs.append(model.wv[t])
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def cos(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0: 
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def coherence_prime(verse_vecs, mu=0.65, sigma=0.15):
    # C: media de cosenos entre versos consecutivos
    if len(verse_vecs) < 2:
        return 0.0
    sims = [cos(verse_vecs[i-1], verse_vecs[i]) for i in range(1, len(verse_vecs))]
    C = float(np.mean(sims)) if sims else 0.0
    # Mapeo a campana con pico en mu
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

# ---------- Load data ----------
file_path = "../data/tcc_ceds_music.csv"
df = pd.read_csv(file_path)

# Filtrado básico
df = df[['artist_name', 'track_name', 'genre', 'lyrics']].dropna(subset=['lyrics', 'genre', 'artist_name'])
df = df[df['lyrics'].str.len() > 0]

# Subconjunto de prueba
# random_state=123 for reproducibility
N_SAMPLE = 500
df_sample = df.sample(N_SAMPLE, random_state=123).reset_index(drop=True)

# ---------- Tokenization per verse ----------
# Representaremos cada canción como lista de versos (cada verso es lista de tokens)
song_verses = []
for txt in tqdm(df_sample['lyrics'].tolist(), desc="Tokenizing"):
    tokens_by_line = tokenize_lines(txt)
    song_verses.append(tokens_by_line)

# Construcción de corpus para Word2Vec: todas las líneas como "oraciones"
sentences = [line for song in song_verses for line in song]

# Si no hay suficiente texto, abortamos
if len(sentences) < 10:
    raise RuntimeError("Muy pocas oraciones para entrenar Word2Vec en el subconjunto.")

# ---------- Train Word2Vec (ligero) ----------
w2v = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=2,
    sg=1,      # skip-gram para captar semántica con poco corpus
    epochs=10
)

# ---------- Compute verse embeddings per song ----------
song_verse_vecs = []
for tokens_by_line in song_verses:
    vvecs = []
    for toks in tokens_by_line:
        v = average_vec(toks, w2v)
        if v is not None:
            vvecs.append(v)
    song_verse_vecs.append(vvecs)

# ---------- Build genre centroids (using song means) ----------
# Centroide de cada canción y luego media por género
df_sample['song_vec'] = [np.mean(vs, axis=0) if len(vs)>0 else np.zeros(w2v.vector_size) for vs in song_verse_vecs]
genre_means = {}
for g, grp in df_sample.groupby('genre'):
    mats = np.vstack(grp['song_vec'].values) if len(grp)>0 else None
    if mats is not None and mats.size > 0:
        m = np.mean(mats, axis=0)
        genre_means[g] = m
    else:
        genre_means[g] = None

# ---------- Compute C', D, N, LQS per song ----------
results = []
for i, row in df_sample.iterrows():
    verses_vecs = song_verse_vecs[i]
    Cp = coherence_prime(verses_vecs)
    D = diversity_score(verses_vecs)
    gcent = genre_means.get(row['genre'])
    N = novelty_score(verses_vecs, gcent)
    LQS = lqs_score(Cp, D, N, w=(0.4, 0.35, 0.25))
    results.append((row['artist_name'], row['genre'], row['track_name'], Cp, D, N, LQS))

res_df = pd.DataFrame(results, columns=['artist_name', 'genre', 'track_name', 'coherence_prime', 'diversity', 'novelty', 'LQS'])

# ---------- Aggregations ----------
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

# Display interactive tables to the user
'''
!pip install caas_jupyter_tools
import caas_jupyter_tools as cj
cj.display_dataframe_to_user("LQS por canción (subset 500)", res_df)
cj.display_dataframe_to_user("Resumen por género", by_genre)
cj.display_dataframe_to_user("Resumen por artista y género", by_artist_genre)

subset_path, genre_path, artist_genre_path
'''