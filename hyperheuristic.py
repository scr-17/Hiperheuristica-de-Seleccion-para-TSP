from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 67
DATA_DIR = Path("data")
LABELS_ORDER = ["2opt_hill_climbing", "mst_2opt", "sa_2opt"]
COST_MAP = {
    "2opt_hill_climbing": "cost_2opt_hill_climbing",
    "sa_2opt": "cost_sa_2opt",
    "mst_2opt": "cost_mst_2opt",
}


def add_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["nn_pairwise_ratio"] = d["mean_nn_dist"] / (d["mean_pairwise_dist"] + 1e-12)
    d["centroid_pairwise_ratio"] = d["mean_centroid_dist"] / (
        d["mean_pairwise_dist"] + 1e-12
    )
    d["expected_mst_scale"] = np.sqrt(d["n"]) * d["mean_nn_dist"]
    return d


def compute_regret(df_eval: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
    min_costs = (
        df_eval[["cost_2opt_hill_climbing", "cost_sa_2opt", "cost_mst_2opt"]]
        .min(axis=1)
        .to_numpy()
    )
    chosen_costs = np.array(
        [df_eval.iloc[i][COST_MAP[pred]] for i, pred in enumerate(y_pred)]
    )
    return chosen_costs - min_costs


def print_class_balance(title: str, series: pd.Series) -> None:
    total = len(series)
    counts = series.value_counts()
    print(f"\n{title}")
    for alg in LABELS_ORDER:
        c = int(counts.get(alg, 0))
        pct = (c / total) * 100.0 if total > 0 else 0.0
        print(f"  {alg:<22}: {c:>3} ({pct:5.2f}%)")


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train_labeled.csv")
    test_df = pd.read_csv(DATA_DIR / "test_labeled.csv")
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print_class_balance(
        "Balance Global de Clases", full_df["best_algorithm"]
    )
    print_class_balance("Balance Entrenamiento", train_df["best_algorithm"])
    print_class_balance("Balance Prueba", test_df["best_algorithm"])

    base_cols = [
        "n",
        "distribution",
        "mean_pairwise_dist",
        "std_coords",
        "mean_centroid_dist",
        "mean_nn_dist",
    ]
    num_cols = [
        "n",
        "mean_pairwise_dist",
        "std_coords",
        "mean_centroid_dist",
        "mean_nn_dist",
        "nn_pairwise_ratio",
        "centroid_pairwise_ratio",
        "expected_mst_scale",
    ]

    X_train = train_df[base_cols]
    y_train = train_df["best_algorithm"]
    X_test = test_df[base_cols]
    y_test = test_df["best_algorithm"]

    majority_class = y_train.value_counts().idxmax()
    y_pred_baseline = np.full(len(y_test), majority_class)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    regret_baseline = compute_regret(test_df, y_pred_baseline)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False), ["distribution"]),
            ("num", StandardScaler(), num_cols),
        ]
    )

    pipeline = Pipeline(
        [
            ("fe", FunctionTransformer(add_geometric_features)),
            ("prep", preprocessor),
            ("clf", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_
    y_pred_knn = best_knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    regret_knn = compute_regret(test_df, y_pred_knn)
    cm_knn = confusion_matrix(y_test, y_pred_knn, labels=LABELS_ORDER)

    print("\nHiperparametros Óptimos KNN:")
    print(f"  Mejores parametros : {grid_search.best_params_}")
    print(f"  Accuracy CV (train): {grid_search.best_score_ * 100:.2f}%")

    print("\nEvaluación en conjunto de prueba (n = 149)")
    print(f"  Baseline trivial ({majority_class}):")
    print(
        f"    Accuracy         : {acc_baseline * 100:.2f}% ({int((y_pred_baseline == y_test).sum())}/{len(y_test)})"
    )
    print(f"    Regret promedio  : {regret_baseline.mean():.6f}")
    print(f"    Regret maximo    : {regret_baseline.max():.6f}")
    print(f"    Regret acumulado : {regret_baseline.sum():.6f}")

    print("\n  Clasificador KNN:")
    print(
        f"    Accuracy         : {acc_knn * 100:.2f}% ({int((y_pred_knn == y_test).sum())}/{len(y_test)})"
    )
    print(
        f"    Diferencia vs Base: {(acc_knn - acc_baseline) * 100:+.2f} puntos porcentuales"
    )
    print(
        f"    Regret promedio  : {regret_knn.mean():.6f} ({((regret_knn.mean() - regret_baseline.mean()) / regret_baseline.mean()) * 100:+.2f}% vs Baseline)"
    )
    print(f"    Regret maximo    : {regret_knn.max():.6f}")
    print(f"    Regret acumulado : {regret_knn.sum():.6f}")
    print(
        f"    Instancias sin regret (regret == 0): {int((regret_knn <= 1e-12).sum())}/{len(y_test)}"
    )

    print("\nMatriz de confusión en prueba (Filas = Real, Columnas = Predicho)")
    cm_df = pd.DataFrame(
        cm_knn,
        index=[f"Real: {lbl}" for lbl in LABELS_ORDER],
        columns=[f"Pred: {lbl}" for lbl in LABELS_ORDER],
    )
    print(cm_df.to_string())

    regret_detail = pd.DataFrame(
        {
            "instance_id": test_df["instance_id"],
            "real": y_test,
            "pred_knn": y_pred_knn,
            "regret_knn": regret_knn,
            "regret_baseline": regret_baseline,
        }
    )
    regret_detail.to_csv(DATA_DIR / "test_regret_knn.csv", index=False)


if __name__ == "__main__":
    main()
