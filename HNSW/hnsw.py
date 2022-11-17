from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from random import random
from typing import List, Optional, Set, Tuple
import numpy as np

from HNSW.distance import DistanceFunctionType, l2_distance

class HNSW(object):

    def __init__(self, m : int =5, ef : int =200, m0 : Optional[int] = None, distance_func: DistanceFunctionType = l2_distance):
        self.data : List[np.ndarray] = []

        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._entry_point : Optional[int] = None
        self._distance_func = distance_func

    def __getitem__(self, idx):
        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return

    def add(self, elem : np.ndarray):
        distance_func = self._distance_func
        data = self.data
        graphs = self._graphs
        point = self._entry_point
        m = self._m
        ef = self._ef

        level = self._gen_level()

        idx = len(data)
        data.append(elem)

        # HNSW contains points
        if point is not None:
            dist = distance_func(elem, data[point])

            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)

            ep = [(-dist, point)]

            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0

                ep = self._search_graph(elem, ep, layer, ef)

                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)


                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)

        for i in range(len(graphs), level):
            graphs.append({
                idx: {}
            })
            self._entry_point = idx
                
    def search_brute(self, q : np.ndarray, k : Optional[int] = None):
        distances : List[Tuple[int, np.floating]] = []
        data = self.data
        distance_func = self._distance_func
        for i, p in enumerate(data):
            dist = distance_func(q, p)
            distances.append((i, dist))

        distances.sort(key=lambda el: el[1])
        return distances if k is None else distances[:k]

    def search(self, q, k=None, ef=None):
        """Find k closest points to q"""
        distance_func = self._distance_func
        graphs = self._graphs
        point = self._entry_point

        ef = self._ef if ef is None else ef

        dist = distance_func(q, self.data[point])

        # Look in top 2 layers
        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)

        # Look at bottom level
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -md) for md, idx in ep]

    def _gen_level(self):
        return int(
            -log2(random()) * self._level_mult
        ) + 1

    # Search graph when ef = 1 (entry constraint?)
    def _search_graph_ef1(self, q, entry, dist, layer):
        distance_func = self._distance_func
        data = self.data
        

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break

            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = [distance_func(q, data[e]) for e in edges]
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))

        return best, best_dist

    def _search_graph(self, q, ep, layer, ef):
        distance_func = self._distance_func
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break

            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = [distance_func(q, data[e]) for e in edges]
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _select(self, d, to_insert, m, g, heap=False):
        nb_dicts = [g[idx] for idx in d]

        def prioritize(idx, dist):
            return any(nd.get(idx, float('inf')) < dist for nd in nb_dicts), dist,  idx

        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, 
                (prioritize(idx, -mdist) for mdist, idx in to_insert)
            )
        
        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)

        unchecked = m - len(d)
        assert 0 <= unchecked <= m

        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)

        if to_check > 0:
            checked_del = nlargest(to_check, (
                prioritize(idx, dist) for idx, dist in d.items()
            ))
        else:
            checked_del = []

        for _, dist, idx in to_insert:
            d[idx] = dist

        zipped = zip(checked_ins, checked_del)
        
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break

            del d[idx_old]
            d[idx_new] = d_new
            assert len(d) == m