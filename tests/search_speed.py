from math import floor
from typing import Dict
from HNSW.hnsw import HNSW
from data import get_random_vectors
from time import time
import itertools
from tabulate import tabulate
from tqdm import tqdm
from pprint import pprint

num_points = 1_000
dimension = 2

hnsw = HNSW()

stats = [["n", "k", "HNSW", "Brute", "X Speed"]]

search_vec = get_random_vectors(n=1, dim=dimension, from_file=False)[0]

vectors = []

for i in tqdm(range(0, 10)):
    new_vecs = get_random_vectors(n=num_points, dim=dimension, from_file=False)
    [hnsw.add(v) for v in new_vecs]
    vectors.extend(new_vecs)

    for k in [1, 10, 20]:
        stat = [len(vectors), k]
        
        start = time()
        points_hnsw = [(vectors[idx], dist) for idx, dist in hnsw.search(search_vec, k=1)]
        dur = time() - start
        stat.append(dur * 1000)

        start = time()
        points_brute = [(vectors[idx], dist) for idx, dist in hnsw.search_brute(search_vec, k=1)]
        dur = time() - start
        stat.append(dur * 1000)

        stat.append(stat[3]/stat[2])

        stats.append(stat)

print(tabulate(stats))
  