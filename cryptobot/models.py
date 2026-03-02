"""Mixin para model training: entrenamiento, optimización y feature importance."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .constants import COLOR_PALETTE


class ModelsMixin:
    """Métodos de model training: train_models(), optimize_model(), feature_importance(), plot_feature_importance()."""

    def train_models(
        self,
        window: str = "expanding",
        window_size: int = 60,
    ) -> "ModelsMixin":
        """
        Entrena y compara múltiples modelos ML para la estrategia seleccionada.

        Modelos evaluados:
        - Logistic Regression
        - SVM (RBF kernel)
        - Random Forest
        - XGBoost

        Usa TimeSeriesSplit para validación temporal (sin data leakage).

        Parameters
        ----------
        window : str, default "expanding"
            Tipo de ventana de entrenamiento:
            - "expanding": usa todos los datos disponibles hasta t.
              Más estable, más datos de training.
            - "sliding": usa solo los últimos window_size períodos.
              Se adapta mejor a cambios de régimen.
        window_size : int, default 60
            Tamaño de la ventana para modo "sliding" (en períodos).

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        El mejor modelo se auto-selecciona basado en F1-score.
        Resultados de comparación disponibles en self.model_comparison.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado select_strategy() previamente.
        """
        self._require_features()
        self._require_strategy()

        df = self.features.copy()

        # ── 1. Crear target binario según estrategia ───────
        forward_return = df["Close"].pct_change().shift(-1)

        if self.selected_strategy == "trend_following":
            # Target = 1 cuando el precio sube en el siguiente período
            df["target"] = (forward_return > 0).astype(int)

        elif self.selected_strategy == "mean_reversion":
            # Target = 1 cuando precio se desvía de SMA_20 y el siguiente
            # movimiento va hacia la media
            sma = df["Close"].rolling(20).mean()
            deviation = df["Close"] - sma
            # Si está por encima de la media y baja, o por debajo y sube → revierte
            df["target"] = (
                ((deviation > 0) & (forward_return < 0))
                | ((deviation < 0) & (forward_return > 0))
            ).astype(int)

        elif self.selected_strategy == "momentum":
            # Target = 1 cuando el retorno del siguiente período tiene
            # el mismo signo que el actual
            current_return = df["Close"].pct_change()
            df["target"] = (
                (forward_return > 0) & (current_return > 0)
                | (forward_return < 0) & (current_return < 0)
            ).astype(int)

        # Eliminar filas sin target (última fila por shift(-1))
        df.dropna(subset=["target"], inplace=True)
        df["target"] = df["target"].astype(int)

        # ── 2. Preparar X e y ─────────────────────────────
        exclude_cols = {"Open", "High", "Low", "Close", "Volume", "regime", "target", "returns"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self._feature_cols = feature_cols

        X = df[feature_cols].values
        y = df["target"].values

        # ── 3. Split temporal ──────────────────────────────
        tscv = TimeSeriesSplit(n_splits=5)

        # ── 4. Definir modelos ─────────────────────────────
        models = {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]),
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC()),
            ]),
            "random_forest": Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]),
            "xgboost": Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBClassifier(
                    n_estimators=100,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                )),
            ]),
        }

        # ── 5. Entrenar y evaluar cada modelo (out-of-fold) ──
        results = {}
        for name, pipeline in models.items():
            all_y_true = []
            all_y_pred = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Sliding window: recortar train a últimos window_size registros
                if window == "sliding" and len(X_train) > window_size:
                    X_train = X_train[-window_size:]
                    y_train = y_train[-window_size:]

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)

            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)

            results[name] = {
                "accuracy": accuracy_score(all_y_true, all_y_pred),
                "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
                "recall": recall_score(all_y_true, all_y_pred, zero_division=0),
                "f1": f1_score(all_y_true, all_y_pred, zero_division=0),
            }

        # ── 6. Seleccionar mejor modelo por F1 ────────────
        best_name = max(results, key=lambda k: results[k]["f1"])

        # ── 7. Refit mejor modelo en datos completos ──────
        best_pipeline = models[best_name]
        best_pipeline.fit(X, y)

        # ── 8. Guardar estado ──────────────────────────────
        self.model = best_pipeline
        self.model_name = best_name
        self.model_metrics = results[best_name]
        self.model_comparison = pd.DataFrame(results).T
        self.model_comparison.index.name = "model"
        self._X_train = X
        self._y_train = y

        # ── 9. Print tabla comparativa ─────────────────────
        print("=" * 75)
        print("🤖 COMPARACIÓN DE MODELOS")
        print("=" * 75)
        print(f"\n{'Modelo':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 65)

        for name, metrics in results.items():
            star = " ⭐" if name == best_name else ""
            print(
                f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}{star}"
            )

        print(f"\n✅ Mejor modelo: {best_name} (F1: {results[best_name]['f1']:.4f})")

        return self

    def optimize_model(self, model_name: Optional[str] = None) -> "ModelsMixin":
        """
        Optimiza hiperparámetros del modelo usando GridSearchCV.

        Parameters
        ----------
        model_name : str, optional
            Modelo a optimizar. Si None, optimiza el mejor modelo
            de train_models(). Opciones: "logistic_regression",
            "svm", "random_forest", "xgboost".

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        Usa TimeSeriesSplit como cv para respetar temporalidad.
        Los grids de hiperparámetros están predefinidos por modelo.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado train_models() previamente.
        """
        self._require_model()

        if model_name is None:
            model_name = self.model_name

        # ── 1. Métricas antes de optimizar ─────────────────
        old_metrics = self.model_metrics.copy()

        # ── 2. Crear pipeline fresco para GridSearch ───────
        estimators = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "svm": SVC(),
            "random_forest": RandomForestClassifier(random_state=42),
            "xgboost": XGBClassifier(
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            ),
        }

        param_grids = {
            "logistic_regression": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear", "saga"],
            },
            "svm": {
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
                "model__kernel": ["rbf", "linear"],
            },
            "random_forest": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 10, None],
                "model__min_samples_split": [2, 5, 10],
            },
            "xgboost": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.3],
            },
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimators[model_name]),
        ])

        # ── 3. GridSearchCV con TimeSeriesSplit ────────────
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=TimeSeriesSplit(n_splits=5),
            scoring="f1",
            n_jobs=-1,
        )
        grid_search.fit(self._X_train, self._y_train)

        # ── 4. Actualizar modelo y métricas ────────────────
        self.model = grid_search.best_estimator_

        # Calcular nuevas métricas con cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        all_y_true = []
        all_y_pred = []

        for train_idx, test_idx in tscv.split(self._X_train):
            X_train = self._X_train[train_idx]
            X_test = self._X_train[test_idx]
            y_train = self._y_train[train_idx]
            y_test = self._y_train[test_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        new_metrics = {
            "accuracy": accuracy_score(all_y_true, all_y_pred),
            "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
            "recall": recall_score(all_y_true, all_y_pred, zero_division=0),
            "f1": f1_score(all_y_true, all_y_pred, zero_division=0),
        }

        # Refit en datos completos
        self.model.fit(self._X_train, self._y_train)
        self.model_metrics = new_metrics

        # ── 5. Print resultados ────────────────────────────
        print("=" * 65)
        print(f"🔧 OPTIMIZACIÓN — {model_name}")
        print("=" * 65)
        print(f"\nMejores parámetros: {grid_search.best_params_}")
        print(f"\n{'Métrica':<15} {'Antes':>10} {'Después':>10} {'Cambio':>10}")
        print("-" * 45)

        for metric in ["accuracy", "precision", "recall", "f1"]:
            old_val = old_metrics[metric]
            new_val = new_metrics[metric]
            diff = new_val - old_val
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"{metric:<15} {old_val:>9.4f} {new_val:>9.4f} {arrow} {abs(diff):.4f}")

        return self

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Muestra las features más importantes del modelo entrenado.

        Parameters
        ----------
        top_n : int, default 15
            Número de features a mostrar.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas: feature, importance, ordenado descendente.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado train_models() previamente.
        """
        self._require_model()

        estimator = self.model.named_steps["model"]

        # Tree-based: feature_importances_
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_

        # Linear models: coef_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_[0])

        # SVM / otros: permutation_importance
        else:
            result = permutation_importance(
                self.model,
                self._X_train,
                self._y_train,
                scoring="f1",
                n_repeats=10,
                random_state=42,
            )
            importances = result.importances_mean

        df_imp = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": importances,
        })
        df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)

        return df_imp.head(top_n)

    def plot_feature_importance(self, top_n: int = 15) -> None:
        """
        Gráfico de barras horizontal con las features más importantes.

        Parameters
        ----------
        top_n : int, default 15
            Número de features a mostrar.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado train_models() previamente.
        """
        self._require_model()

        df_imp = self.feature_importance(top_n)

        # Invertir orden para que el más importante quede arriba
        df_imp = df_imp.iloc[::-1]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_imp["importance"],
            y=df_imp["feature"],
            orientation="h",
            marker_color=COLOR_PALETTE["yellow"],
        ))

        fig.update_layout(
            title=f"Top {top_n} Features — {self.model_name}",
            xaxis_title="Importancia",
            yaxis_title="Feature",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            height=max(400, top_n * 30),
        )

        fig.show()
