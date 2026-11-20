"""Utilidades y heurísticas de bajo nivel para el Problema del Viajante (TSP).

Todas las funciones trabajan con una matriz de distancias ``dist`` de tamaño
``n x n`` y con recorridos representados como permutaciones de ``0..n-1``.
"""

import math
import random


def build_dist_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    """Construye la matriz de distancias euclidianas entre todos los puntos."""
    n = len(points)
    return [[math.dist(points[i], points[j]) for j in range(n)] for i in range(n)]


def get_tour_length(dist: list[list[float]], tour: list[int]) -> float:
    """Calcula la longitud total de un recorrido cíclico."""
    n = len(tour)
    if n == 0:
        return 0.0
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def mst_prim(dist: list[list[float]]) -> dict[int, list[int]]:
    """Calcula un Árbol de Expansión Mínima (Prim) y devuelve sus hijos.

    La raíz del árbol es el vértice ``0``. El resultado se representa como un
    diccionario ``padre -> lista_de_hijos`` para poder recorrerlo en preorden.
    """
    n = len(dist)
    if n == 0:
        return {}

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
    """Devuelve los vértices del árbol en recorrido preorden."""
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def local_search_2opt(dist: list[list[float]], tour: list[int]) -> list[int]:
    """Mejora un recorrido con búsqueda local 2-opt (primera mejora).

    En cada paso se invierte el segmento ``tour[i:k+1]`` si con ello se reducen
    las dos aristas afectadas. Se repite hasta alcanzar un óptimo local.
    """
    t = tour.copy()
    n = len(t)
    if n < 4:
        return t

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a = t[(i - 1) % n]
            b = t[i]
            d_ab = dist[a][b]
            for k in range(i + 1, n):
                # Invertir todo el recorrido no cambia el ciclo resultante.
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
    dist: list[list[float]],
    initial_tour: list[int],
    rng: random.Random,
    temperatures: dict[str, float | int],
) -> list[int]:
    """Recocido simulado sobre el vecindario 2-opt.

    Acepta soluciones peores con probabilidad ``exp(-delta / T)`` y enfría la
    temperatura multiplicándola por ``SA_COOLING_RATE`` en cada iteración.
    """
    curr = initial_tour.copy()
    n = len(curr)
    if n < 4:
        return curr

    temp = float(temperatures["SA_T0"])
    t_min = float(temperatures["SA_T_MIN"])
    cooling_rate = float(temperatures["SA_COOLING_RATE"])
    max_iter = int(temperatures["SA_MAX_ITER"])

    for _ in range(max_iter):
        if temp <= t_min:
            break

        i, k = sorted(rng.sample(range(n), 2))
        # Invertir el recorrido completo no produce un ciclo distinto.
        if not (i == 0 and k == n - 1):
            a, b = curr[(i - 1) % n], curr[i]
            c, d = curr[k], curr[(k + 1) % n]
            delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

            if delta < -1e-12 or rng.random() < math.exp(-max(delta, 0.0) / temp):
                curr[i : k + 1] = curr[i : k + 1][::-1]

        temp *= cooling_rate

    return curr
