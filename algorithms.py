import math

def build_dist_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    n = len(points)
    return [[math.dist(points[i], points[j]) for j in range(n)] for i in range(n)]


def get_tour_length(dist: list[list[float]], tour: list[int]) -> float:
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def mst_prim(dist: list[list[float]]) -> dict[int, list[int]]:
    n = len(dist)
    best = {v: (dist[0][v], 0) for v in range(1, n)}
    children = {v: [] for v in range(n)}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            d = dist[v][u]
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def local_search_2opt(dist: list[list[float]], tour: list[int]) -> list[int]:
    t = tour.copy()
    n = len(t)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a = t[(i - 1) % n]
            b = t[i]
            d_ab = dist[a][b]
            for k in range(i + 1, n):
                if i == 0 and k == n - 1:
                    continue
                c = t[k]
                d = t[(k + 1) % n]
                if dist[a][c] + dist[b][d] < d_ab + dist[c][d] - 1e-12:
                    t[i : k + 1] = t[i : k + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return t


def simulated_annealing(
        dist: list[list[float]], initial_tour: list[int], rng: random.Random, temperatures: dict[str, float | int]
) -> list[int]:
    curr = initial_tour.copy()
    n = len(curr)
    curr_len = get_tour_length(dist, curr)
    temp = temperatures["SA_T0"]

    for _ in range(temperatures["SA_MAX_ITER"]):
        if temp <= temperatures["SA_T_MIN"]:
            break
        i, k = sorted(rng.sample(range(n), 2))
        if i == 0 and k == n - 1:
            delta = 0.0
        else:
            a, b = curr[(i - 1) % n], curr[i]
            c, d = curr[k], curr[(k + 1) % n]
            delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

        if delta < -1e-12 or rng.random() < math.exp(-max(delta, 0.0) / temp):
            curr[i : k + 1] = curr[i : k + 1][::-1]
            curr_len += delta

        temp *= temperatures["SA_COOLING_RATE"]

    return curr


