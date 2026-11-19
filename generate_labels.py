import csv
import random
import time
import algorithms as ag 
from collections import Counter
from pathlib import Path

SEED = 67
DATA_DIR = Path("data")
INSTANCES_DIR = DATA_DIR / "instances"

temperatures = {
    "SA_T0" : 100.0,
    "SA_T_MIN" : 0.001,
    "SA_COOLING_RATE" : 0.99,
    "SA_MAX_ITER" : 1000
}

def load_instance(instance_id: str) -> list[tuple[float, float]]:
    filepath = INSTANCES_DIR / f"{instance_id}.csv"
    with filepath.open("r", newline="", encoding="utf-8") as f:
        return [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]


def evaluate_instance(
    points: list[tuple[float, float]], rng: random.Random
) -> tuple[float, float, float, str]:
    dist = ag.build_dist_matrix(points)
    n = len(points)
    random_tour = rng.sample(range(n), n)

    cost_2opt_hc = ag.get_tour_length(dist, ag.local_search_2opt(dist, random_tour))

    cost_sa_2opt = ag.get_tour_length(
        dist, ag.simulated_annealing(dist, random_tour, rng, temperatures)
    )

    tour_mst = ag.preorder(ag.mst_prim(dist))
    cost_mst_2opt = ag.get_tour_length(dist, ag.local_search_2opt(dist, tour_mst))

    costs = {
        "2opt_hill_climbing": cost_2opt_hc,
        "sa_2opt": cost_sa_2opt,
        "mst_2opt": cost_mst_2opt,
    }
    best_algorithm = min(costs, key=costs.get)

    return cost_2opt_hc, cost_sa_2opt, cost_mst_2opt, best_algorithm

def process_split(input_csv: Path, output_csv: Path, rng: random.Random) -> Counter:
    counts = Counter()
    with input_csv.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames) + [
            "cost_2opt_hill_climbing",
            "cost_sa_2opt",
            "cost_mst_2opt",
            "best_algorithm",
        ]
        rows_out = []

        for row in reader:
            points = load_instance(row["instance_id"])
            c_hc, c_sa, c_mst, best_alg = evaluate_instance(points, rng)

            row["cost_2opt_hill_climbing"] = f"{c_hc:.6f}"
            row["cost_sa_2opt"] = f"{c_sa:.6f}"
            row["cost_mst_2opt"] = f"{c_mst:.6f}"
            row["best_algorithm"] = best_alg

            rows_out.append(row)
            counts[best_alg] += 1

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return counts


def print_balance(title: str, counts: Counter) -> None:
    total = sum(counts.values())
    print(f"\n=== {title} (n = {total}) ===")
    for alg in ["2opt_hill_climbing", "mst_2opt", "sa_2opt"]:
        c = counts[alg]
        pct = (c / total * 100) if total > 0 else 0.0
        print(f"  {alg:<22}: {c:>3} ({pct:5.2f}%)")


if __name__ == "__main__":
    t_start = time.perf_counter()
    rng = random.Random(SEED)

    train_in = DATA_DIR / "train.csv"
    test_in = DATA_DIR / "test.csv"
    train_out = DATA_DIR / "train_labeled.csv"
    test_out = DATA_DIR / "test_labeled.csv"

    print(f"SEED = {SEED}")
    train_counts = process_split(train_in, train_out, rng)
    test_counts = process_split(test_in, test_out, rng)
    total_counts = train_counts + test_counts

    print_balance("BALANCE ENTRENAMIENTO (train_labeled.csv)", train_counts)
    print_balance("BALANCE PRUEBA (test_labeled.csv)", test_counts)
    print_balance("BALANCE GLOBAL (500 instancias)", total_counts)

