# Hiperheurística de Selección para TSP

Implementación de una **hiperheurística de selección** para el Problema del
Viajante (TSP, *Traveling Salesman Problem*). Dada una instancia nueva, un
clasificador **KNN** predice cuál de tres heurísticas de bajo nivel conviene
ejecutar para obtener el recorrido de menor costo, en lugar de ejecutar las
tres y quedarse con la mejor.

El proyecto incluye la generación automática de etiquetas, el entrenamiento y
la evaluación del clasificador, y mide el desempeño tanto en **accuracy** como
en **regret** (cuánto se pierde por elegir una heurística que no era la óptima).

---

## 1. Objetivo

Para cada instancia de TSP se busca un ciclo hamiltoniano de longitud mínima
sobre puntos en el plano euclidiano. Resolver el problema de forma exacta es
costoso, por lo que se usan heurísticas. La pregunta que responde este proyecto
es:

> ¿Es posible predecir, a partir de características geométricas de la
> instancia, cuál heurística de bajo nivel dará el mejor resultado?

El flujo es:

```
Instancias TSP → ejecutar 3 heurísticas → etiquetar con la mejor
      → entrenar KNN con características geométricas
      → predecir la heurística para instancias nuevas
      → evaluar accuracy y regret
```

---

## 2. Estructura del repositorio

```
.
├── exec.sh                      # Ejecuta todo el flujo (venv + dependencias + scripts)
├── requirements.txt             # Dependencias de Python con versiones fijas
├── README.md
├── data/
│   ├── instances/               # 500 instancias (inst_0001.csv, ...)
│   ├── train.csv                # 351 instancias de entrenamiento
│   ├── test.csv                 # 149 instancias de prueba
│   ├── train_labeled.csv        # (generado) train.csv + costos y etiqueta
│   ├── test_labeled.csv         # (generado) test.csv + costos y etiqueta
│   └── test_regret_knn.csv      # (generado) detalle del regret por instancia
└── src/
    ├── __init__.py
    ├── algorithms.py            # Matriz de distancias, 2-opt, SA, MST/Prim
    ├── generate_labels.py       # Ejecuta las heurísticas y etiqueta cada instancia
    └── hyperheuristic.py        # Entrena el KNN y evalúa accuracy/regret
```

Los archivos `*_labeled.csv` y `test_regret_knn.csv` se generan al ejecutar el
proyecto y no necesitan existir de antemano.

---

## 3. Datos

- **500 instancias** en total: 351 para entrenamiento y 149 para prueba.
- Coordenadas en el cuadrado unitario `[0, 1] × [0, 1]`.
- Tamaños `n ∈ {5, 10, 15, 20, 30, 40, 60, 80, 100}`.
- Tres familias de distribución: `uniform`, `clustered` y `border`.
- Cada instancia está en `data/instances/<instance_id>.csv` con columnas `x, y`.

Los archivos `train.csv` y `test.csv` contienen una fila por instancia con las
siguientes características geométricas:

| Columna               | Descripción                                                        |
| --------------------- | ------------------------------------------------------------------ |
| `instance_id`         | Identificador de la instancia (`inst_0001`, ...).                  |
| `n`                   | Cantidad de puntos/ciudades.                                       |
| `distribution`        | Familia de generación: `uniform`, `clustered` o `border`.          |
| `mean_pairwise_dist`  | Distancia euclidiana promedio entre todos los pares de puntos.     |
| `std_coords`          | Desviación estándar de las coordenadas.                            |
| `mean_centroid_dist`  | Distancia promedio de los puntos al centroide de la instancia.     |
| `mean_nn_dist`        | Distancia promedio al vecino más cercano.                          |

Características derivadas que construye `hyperheuristic.py`:

| Característica             | Fórmula                                              |
| -------------------------- | ---------------------------------------------------- |
| `nn_pairwise_ratio`        | `mean_nn_dist / mean_pairwise_dist`                  |
| `centroid_pairwise_ratio`  | `mean_centroid_dist / mean_pairwise_dist`            |
| `expected_mst_scale`       | `sqrt(n) * mean_nn_dist`                             |

La idea es capturar qué tan concentrados están los puntos, qué tan uniforme es
su separación y qué tan grande se espera que sea el árbol de expansión mínima.

---

## 4. Heurísticas de bajo nivel

Todas trabajan con una matriz de distancias euclidianas y representan un
recorrido como una permutación de `0..n-1`.

### 4.1 `2opt_hill_climbing`

Parte de un recorrido aleatorio y aplica **búsqueda local 2-opt con primera
mejora**: si al invertir el segmento `tour[i:k+1]` se reducen las dos aristas
afectadas, se aplica el cambio y se continúa hasta alcanzar un óptimo local.

Para las aristas `(a, b)` y `(c, d)`, el cambio se acepta cuando:

```
dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d]
```

### 4.2 `sa_2opt`

**Recocido simulado** sobre el vecindario 2-opt, seguido de una refinación con
2-opt. Acepta soluciones peores con probabilidad `exp(-Δ / T)` y aplica
enfriamiento geométrico.

Parámetros en `generate_labels.py`:

| Parámetro          | Valor   |
| ------------------ | ------- |
| `SA_T0`            | `100.0` |
| `SA_T_MIN`         | `0.001` |
| `SA_COOLING_RATE`  | `0.99`  |
| `SA_MAX_ITER`      | `1000`  |

### 4.3 `mst_2opt`

Construye un **Árbol de Expansión Mínima** con el algoritmo de **Prim**,
recorre el árbol en **preorden** para obtener un recorrido inicial y luego lo
mejora con 2-opt. Es una forma clásica de obtener una solución inicial
razonable para TSP.

### 4.4 Etiquetado

Para cada instancia se ejecutan las tres heurísticas, se calcula el costo de
cada recorrido y se asigna como etiqueta la de menor costo:

```
best_algorithm = argmin(costo_2opt_hc, costo_sa_2opt, costo_mst_2opt)
```

El resultado se guarda en `data/train_labeled.csv` y `data/test_labeled.csv`
junto con las columnas `cost_2opt_hill_climbing`, `cost_sa_2opt`,
`cost_mst_2opt` y `best_algorithm`.

---

## 5. Hiperheurística de selección

El modelo implementado es un **KNN** dentro de un `Pipeline` de scikit-learn:

1. **Ingeniería de características**: agrega las razones y la escala esperada
   del MST.
2. **Preprocesamiento**:
   - `OneHotEncoder` para `distribution`.
   - `StandardScaler` para todas las variables numéricas.
3. **Clasificador**: `KNeighborsClassifier`.

La búsqueda de hiperparámetros usa `GridSearchCV` con validación cruzada
estratificada de 4 folds (`StratifiedKFold`, `shuffle=True`, semilla 67):

| Hiperparámetro | Valores probados                                      |
| -------------- | ----------------------------------------------------- |
| `n_neighbors`  | `3, 5, 7, 9, 11, 13, 15, 17, 19, 21`                  |
| `weights`      | `uniform`, `distance`                                 |
| `metric`       | `euclidean`, `manhattan`                              |

### Métricas

- **Accuracy**: porcentaje de instancias en las que se predijo la heurística
  que realmente era la mejor.
- **Baseline trivial**: siempre predecir la clase mayoritaria de entrenamiento.
- **Regret** por instancia:

```
regret = costo_heuristica_elegida - min(costo_2opt_hc, costo_sa_2opt, costo_mst_2opt)
```

Un regret de `0` significa que la heurística elegida fue la mejor (o empató).
Se reportan regret promedio, máximo y acumulado, además de la matriz de
confusión.

---

## 6. Requisitos

- **Python 3.12 o superior** (las versiones fijadas de NumPy lo requieren).
- `bash` y `pip`.
- Dependencias de `requirements.txt`:

```
numpy==2.5.5
pandas==3.1.1
scikit-learn==1.9.1
```

`exec.sh` crea automáticamente el entorno virtual `.venv`, instala las
dependencias y ejecuta los dos scripts en orden.

---

## 7. Uso

### Ejecución completa (recomendada)

```bash
./exec.sh
```

El script:

1. Verifica que exista Python 3.12+.
2. Crea `.venv` si no existe (o reutiliza el existente).
3. Instala `requirements.txt`.
4. Ejecuta `python -m src.generate_labels`.
5. Ejecuta `python -m src.hyperheuristic`.

Si se quiere usar otro intérprete:

```bash
PYTHON_BIN=python3.13 ./exec.sh
```

### Ejecución manual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.generate_labels
python -m src.hyperheuristic
```

También se pueden ejecutar los scripts directamente:

```bash
python src/generate_labels.py
python src/hyperheuristic.py
```

Ambos scripts resuelven las rutas de datos a partir de la ubicación del
archivo, por lo que funcionan sin importar el directorio actual.

### Salidas

- `data/train_labeled.csv`, `data/test_labeled.csv`: instancias etiquetadas.
- `data/test_regret_knn.csv`: instancia, etiqueta real, predicción y regret
  del KNN y del baseline.
- Reporte por consola con balances, hiperparámetros, accuracy, regret y matriz
  de confusión.

---

## 8. Resultados observados

Resultados reproducibles con semilla `SEED = 67` y las versiones fijadas en
`requirements.txt` (última ejecución del proyecto).

### Balance de etiquetas

| Heurística            | Train             | Test              | Global            |
| --------------------- | ----------------- | ----------------- | ----------------- |
| `2opt_hill_climbing`  | 194 (55.27 %)     | 84 (56.38 %)      | 278 (55.60 %)     |
| `mst_2opt`            | 66 (18.80 %)      | 27 (18.12 %)      | 93 (18.60 %)      |
| `sa_2opt`             | 91 (25.93 %)      | 38 (25.50 %)      | 129 (25.80 %)     |

`2opt_hill_climbing` sigue siendo la heurística más frecuente (algo más de la
mitad de las instancias), pero `sa_2opt` gana una cuarta parte del total. Las
tres clases tienen presencia suficiente para que el problema de clasificación
no sea trivial.

### Hiperparámetros elegidos por la validación cruzada

```
n_neighbors = 3
weights     = distance
metric      = euclidean
Accuracy CV = 53.85 %
```

### Evaluación en prueba (n = 149)

| Modelo               | Accuracy          | Regret promedio | Regret máximo | Regret acumulado |
| -------------------- | ----------------- | --------------- | ------------- | ---------------- |
| Baseline (mayoría)   | 56.38 % (84/149)  | 0.047625        | 0.614201      | 7.096101         |
| KNN                  | 53.69 % (80/149)  | 0.041893        | 0.568458      | 6.241998         |

El KNN pierde 2.68 puntos porcentuales de accuracy frente al baseline, pero
reduce el regret promedio un 12.04 % (de `0.047625` a `0.041893`). Es decir,
se equivoca algo más seguido, pero cuando se equivoca elige heurísticas más
cercanas a la óptima. Las instancias sin regret son 92 de 149.

### Matriz de confusión (test)

| Real \ Predicho       | `2opt_hill_climbing` | `mst_2opt` | `sa_2opt` |
| --------------------- | -------------------- | ---------- | --------- |
| `2opt_hill_climbing`  | 57                   | 8          | 19        |
| `mst_2opt`            | 10                   | 5          | 12        |
| `sa_2opt`             | 16                   | 4          | 18        |

### Interpretación

- La clase mayoritaria es fuerte, pero ya no domina de forma abrumadora:
  `sa_2opt` gana una de cada cuatro instancias.
- El KNN mejora el **regret** aunque empeore la **accuracy**. Esto muestra que
  la accuracy no es la métrica más alineada con el objetivo real: importa más
  cuánto costo se pierde que acertar la etiqueta exacta.
- El clasificador confunde sobre todo `mst_2opt` y `sa_2opt` con
  `2opt_hill_climbing`; `mst_2opt` es la clase con peor recall (5 de 27).
- Las diferencias de costo entre heurísticas suelen ser pequeñas y las
  etiquetas dependen de recorridos aleatorios, así que hay ruido inherente en
  la clase “mejor heurística”.
- Una mejora natural sería optimizar directamente el **regret esperado**
  (clasificación sensible al costo) en lugar de la accuracy simple.

Estos resultados son parte del análisis del proyecto, no un fallo de
ejecución: el pipeline funciona de principio a fin y deja evidencia del
compromiso entre accuracy y regret en el conjunto de prueba actual.

---

## 9. Reproducibilidad

- Semilla fija `SEED = 67` en la generación de etiquetas y en la validación
  cruzada.
- Las instancias y los recorridos aleatorios se procesan en orden fijo.
- Las versiones de NumPy, pandas y scikit-learn están fijadas en
  `requirements.txt`.

Cambiar las versiones de las librerías o el orden de procesamiento puede
alterar empates y, por lo tanto, los resultados.

---

## 10. Posibles mejoras

- Ajustar la temperatura inicial del recocido simulado a la escala de cada
  instancia (por ejemplo, en función de `mean_pairwise_dist`).
- Reemplazar la accuracy por una función de costo que penalice el regret, o
  usar un modelo que estime el costo esperado de cada heurística.
- Agregar características más informativas: estadísticas de longitudes de
  aristas, longitud del MST, medidas de clustering, estimaciones de cotas
  inferiores.
- Ejecutar cada heurística varias veces y promediar costos para reducir el
  ruido en las etiquetas.
- Probar otros clasificadores (Random Forest, Gradient Boosting) y comparar
  contra el baseline con validación anidada.
- Guardar el modelo entrenado (por ejemplo, con `joblib`) y ofrecer una CLI
  para predecir la heurística recomendada para una instancia nueva.
