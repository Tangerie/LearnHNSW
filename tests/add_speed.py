from math import floor
from HNSW.hnsw import HNSW
from data import get_random_vectors
from time import time
import itertools
from tabulate import tabulate
from tqdm import tqdm

num_points = 1_000


def run_stat_with_config(n, m, dim):
    vectors = get_random_vectors(n=n, dim=dim)
    NUM_FINAL = floor(n * 0.1)
    stat = [n, m, dim]
    hnsw = HNSW(m=m)

    start = time()
    [hnsw.add(v) for v in vectors[:-NUM_FINAL]]
    dur = time() - start

    stat.append(dur)

    start = time()
    [hnsw.add(v) for v in vectors[n - NUM_FINAL:]]
    dur = time() - start

    stat.append(dur)

    stat.append(len(hnsw._graphs))

    return stat


stats = [
    ["n", "m", "dim", "First", "Final", "# Layers"]
]

cases = list(itertools.product(range(5, 26, 5), [32, 64, 128, 256, 512, 1024]))

for m, dim in tqdm(cases, total=len(cases)):
    stats.append(run_stat_with_config(num_points, m, dim))

print(tabulate(stats))