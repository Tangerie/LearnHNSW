from math import floor
from HNSW.hnsw import HNSW
from data import get_random_vectors
from pprint import pprint
from tqdm import tqdm
from time import time

hnsw = HNSW()

num_points = 1_000
dimension = 128

vectors = get_random_vectors(n=num_points, dim=dimension)

NUM_FINAL = floor(num_points * 0.1)

start = time()
[hnsw.add(v) for v in tqdm(vectors[:-NUM_FINAL], total=num_points - NUM_FINAL)]
dur = time() - start

print(f"{len(hnsw.data)} elements added in {dur:.2f}s ({len(hnsw.data)/dur:.2f} el/s)")

start = time()
[hnsw.add(v) for v in tqdm(vectors[num_points - NUM_FINAL:], total=NUM_FINAL)]
dur = time() - start

print(f"{NUM_FINAL} elements added in {dur:.2f}s ({NUM_FINAL/dur:.2f} el/s)")

print("\n")
for i, n in enumerate(hnsw._graphs):
    print(i, len(n))