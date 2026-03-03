"""Mixin para model training: entrenamiento, optimización y feature importance."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import (
    ML_CV_SPLITS,
    ML_N_ESTIMATORS,
    ML_OVERFITTING_THRESHOLD,
    ML_TEST_SIZE,
    ML_WINDOW_SIZE,
    SMA_FAST_WINDOW,
)
from .constants import COLOR_PALETTE


class ModelsMixin:
    """Métodos de model training: train_models(), optimize_model(), feature_importance(), plot_feature_importance()."""

    def train_models(
        self,
        window: str = "expanding",
        window_size: int = ML_WINDOW_SIZE,
        test_size: float = ML_TEST_SIZE,
    ) -> "ModelsMixin":
        """
        Entrena y compara múltiples modelos ML para la estrategia seleccionada.

        Modelos evaluados:
        - Logistic Regression
        - SVM (RBF kernel)
        - Random Forest
        - XGBoost
        - AdaBoost

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
        test_size : float, default 0.2
            Proporción de datos reservados como test set (out-of-sample).
            0.0 = sin split (comportamiento anterior), máximo 0.5.

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

        if not (0 <= test_size <= 0.5):
            raise ValueError(
                f"❌ test_size debe estar entre 0 y 0.5, recibido: {test_size}"
            )

        df = self.features.copy()

        # Limpiar features de estrategias anteriores para evitar contaminación
        _strategy_extra_cols = [
            "spread", "spread_z_score", "spread_z_abs",
            "pair_correlation", "half_life",
            "vol_ratio_custom", "realized_vol_7", "hist_vol_50",
        ]
        extras_to_drop = [c for c in _strategy_extra_cols if c in df.columns]
        if extras_to_drop:
            df.drop(columns=extras_to_drop, inplace=True)
            self.features.drop(columns=extras_to_drop, inplace=True)

        # ── 1. Crear target binario según estrategia ───────
        forward_return = df["Close"].pct_change().shift(-1)

        if self.selected_strategy == "trend_following":
            # Target = 1 cuando el precio sube en el siguiente período
            df["target"] = (forward_return > 0).astype(int)

        elif self.selected_strategy == "mean_reversion":
            # Target = 1 cuando precio se desvía de SMA_20 y el siguiente
            # movimiento va hacia la media
            sma = df["Close"].rolling(SMA_FAST_WINDOW).mean()
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

        elif self.selected_strategy == "breakout":
            # Target = 1 cuando: precio en extremo del canal Donchian +
            # volumen por encima del promedio 20p + follow-through
            required_cols = ["volatility_dcp", "volatility_bbw"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise RuntimeError(
                    f"❌ Breakout requiere mode='full' en create_features(). "
                    f"Columnas faltantes: {missing}"
                )

            # Target direccional: ¿el precio subirá?
            # Los features DCP, BBW y volumen informan al modelo sobre breakouts
            df["target"] = (forward_return > 0).astype(int)

        elif self.selected_strategy == "stat_arb":
            # Requiere datos del par secundario
            self._require_pair_data()

            # Alinear índices entre par primario y secundario
            pair_close = self.pair_data["Close"].reindex(df.index).ffill()
            common_mask = pair_close.notna() & (pair_close > 0) & (df["Close"] > 0)
            pair_close = pair_close[common_mask]
            df = df[common_mask].copy()
            forward_return = df["Close"].pct_change().shift(-1)

            # Spread como log-ratio
            spread = np.log(df["Close"] / pair_close)
            spread_mean = spread.rolling(20).mean()
            spread_std = spread.rolling(20).std()
            z_score = (spread - spread_mean) / spread_std

            # Target direccional: ¿el precio subirá?
            # Los features z_score, spread_z_abs, pair_correlation informan al modelo
            # cuándo las condiciones de mean reversion favorecen comprar vs vender
            df["target"] = (forward_return > 0).astype(int)

            # Agregar features para el modelo
            df["spread"] = spread
            df["spread_z_score"] = z_score
            df["spread_z_abs"] = z_score.abs()
            df["pair_correlation"] = (
                df["Close"].rolling(20).corr(pair_close)
            )
            # Half-life estimación via autocorrelación del spread
            spread_lag = spread.shift(1)
            valid = spread.notna() & spread_lag.notna()
            if valid.sum() > 20:
                beta = np.polyfit(spread_lag[valid].values, spread[valid].values, 1)[0]
                half_life = -np.log(2) / np.log(max(abs(beta), 1e-10)) if abs(beta) < 1 else np.nan
            else:
                half_life = np.nan
            df["half_life"] = half_life

            # Merge features de vuelta a self.features
            stat_arb_features = ["spread", "spread_z_score", "spread_z_abs",
                                 "pair_correlation", "half_life"]
            for col in stat_arb_features:
                self.features[col] = np.nan
                self.features.loc[df.index, col] = df[col]

        elif self.selected_strategy == "volatility":
            # Volatility mean reversion: operar ciclos de expansión/contracción
            returns = df["Close"].pct_change()
            realized_vol_7 = returns.rolling(7).std() * np.sqrt(252)
            hist_vol_50 = returns.rolling(50).std() * np.sqrt(252)

            vol_ratio = realized_vol_7 / hist_vol_50

            # Target direccional: ¿el precio subirá?
            # Los features de vol informan al modelo sobre ciclos de volatilidad
            df["target"] = (forward_return > 0).astype(int)

            # Agregar features para el modelo
            df["vol_ratio_custom"] = vol_ratio
            df["realized_vol_7"] = realized_vol_7
            df["hist_vol_50"] = hist_vol_50

            # Merge features de vuelta a self.features
            vol_features = ["vol_ratio_custom", "realized_vol_7", "hist_vol_50"]
            for col in vol_features:
                self.features[col] = np.nan
                self.features.loc[df.index, col] = df[col]

        # Eliminar filas sin target (última fila por shift(-1))
        df.dropna(subset=["target"], inplace=True)
        df["target"] = df["target"].astype(int)

        # ── 2. Preparar X e y ─────────────────────────────
        exclude_cols = {"Open", "High", "Low", "Close", "Volume", "regime", "target", "returns"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self._feature_cols = feature_cols

        # Limpiar NaN de warmup en features (rolling windows de nuevas estrategias)
        df.dropna(subset=feature_cols, inplace=True)

        X = df[feature_cols].values
        y = df["target"].values

        # ── 3. Split temporal train/test ─────────────────────
        import warnings as _w

        if test_size > 0:
            split_idx = int(len(X) * (1 - test_size))
            if split_idx < 30:
                _w.warn(
                    f"⚠️ Solo {split_idx} registros para train (mínimo 30). "
                    f"Usando test_size=0 (sin split).",
                    stacklevel=2,
                )
                test_size = 0

        if test_size > 0:
            X_train_full = X[:split_idx]
            y_train_full = y[:split_idx]
            X_test_oos = X[split_idx:]
            y_test_oos = y[split_idx:]
            self._test_start = df.index[split_idx]
            self._X_test = X_test_oos
            self._y_test = y_test_oos
        else:
            X_train_full = X
            y_train_full = y
            self._test_start = None
            self._X_test = None
            self._y_test = None

        tscv = TimeSeriesSplit(n_splits=ML_CV_SPLITS)

        # ── 4. Definir modelos ─────────────────────────────
        models = {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]),
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True)),
            ]),
            "random_forest": Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=ML_N_ESTIMATORS, random_state=42)),
            ]),
            "xgboost": Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBClassifier(
                    n_estimators=ML_N_ESTIMATORS,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                )),
            ]),
            "adaboost": Pipeline([
                ("scaler", StandardScaler()),
                ("model", AdaBoostClassifier(n_estimators=ML_N_ESTIMATORS, random_state=42)),
            ]),
        }

        # ── 5. Entrenar y evaluar cada modelo (out-of-fold) ──
        results = {}
        for name, pipeline in models.items():
            all_y_true = []
            all_y_pred = []

            for train_idx, val_idx in tscv.split(X_train_full):
                X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
                y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

                # Sliding window: recortar train a últimos window_size registros
                if window == "sliding" and len(X_tr) > window_size:
                    X_tr = X_tr[-window_size:]
                    y_tr = y_tr[-window_size:]

                # Skip fold si solo hay una clase en training
                if len(np.unique(y_tr)) < 2:
                    continue

                pipeline.fit(X_tr, y_tr)
                y_pred = pipeline.predict(X_val)

                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred)

            # Si todos los folds fueron skipped, métricas = 0
            if len(all_y_true) == 0:
                results[name] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
                continue

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

        # ── 7. Refit mejor modelo en datos de training ────
        best_pipeline = models[best_name]
        best_pipeline.fit(X_train_full, y_train_full)

        # ── 8. Guardar estado ──────────────────────────────
        self.model = best_pipeline
        self.model_name = best_name
        self.model_metrics = results[best_name]
        self.model_comparison = pd.DataFrame(results).T
        self.model_comparison.index.name = "model"
        self._X_train = X_train_full
        self._y_train = y_train_full

        # ── 9. Evaluación Out-of-Sample ────────────────────
        if test_size > 0 and self._X_test is not None and len(self._X_test) > 0:
            y_pred_oos = best_pipeline.predict(self._X_test)
            self._oos_metrics = {
                "accuracy": accuracy_score(self._y_test, y_pred_oos),
                "precision": precision_score(self._y_test, y_pred_oos, zero_division=0),
                "recall": recall_score(self._y_test, y_pred_oos, zero_division=0),
                "f1": f1_score(self._y_test, y_pred_oos, zero_division=0),
            }
        else:
            self._oos_metrics = None

        # ── 10. Print tabla comparativa ────────────────────
        print("=" * 75)
        print("🤖 COMPARACIÓN DE MODELOS")
        print("=" * 75)
        print(f"\n{'Modelo':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 72)

        for name, metrics in results.items():
            star = " ⭐" if name == best_name else ""
            print(
                f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}{star}"
            )

        print(f"\n✅ Mejor modelo: {best_name} (F1: {results[best_name]['f1']:.4f})")

        # ── 11. Print info del split temporal ──────────────
        if test_size > 0 and self._test_start is not None:
            n_train = len(X_train_full)
            n_test = len(self._X_test)
            print(f"\n{'─' * 75}")
            print(f"📊 SPLIT TEMPORAL")
            print(f"{'─' * 75}")
            print(f"  Train: {n_train} registros (hasta {self._test_start.strftime('%Y-%m-%d')})")
            print(f"  Test:  {n_test} registros (desde {self._test_start.strftime('%Y-%m-%d')})")
            print(f"  Ratio: {test_size:.0%} test")

            print(f"\n  {'Métrica':<15} {'CV (train)':>12} {'OOS (test)':>12} {'Ratio':>8}")
            print(f"  {'─' * 50}")

            cv_metrics = results[best_name]
            for metric in ["accuracy", "precision", "recall", "f1"]:
                cv_val = cv_metrics[metric]
                oos_val = self._oos_metrics[metric]
                ratio = oos_val / cv_val if cv_val > 0 else 0
                print(f"  {metric:<15} {cv_val:>12.4f} {oos_val:>12.4f} {ratio:>7.0%}")

            # Warning de overfitting
            cv_f1 = cv_metrics["f1"]
            oos_f1 = self._oos_metrics["f1"]
            if cv_f1 > 0 and oos_f1 < ML_OVERFITTING_THRESHOLD * cv_f1:
                print(f"\n  ⚠️  ALERTA: F1 OOS ({oos_f1:.4f}) < 70% del F1 CV ({cv_f1:.4f})")
                print(f"  ⚠️  Posible overfitting. Considerar: más datos, menos features, regularización.")

        return self

    def optimize_model(self, model_name: Optional[str] = None) -> "ModelsMixin":
        """
        Optimiza hiperparámetros del modelo usando GridSearchCV.

        Parameters
        ----------
        model_name : str, optional
            Modelo a optimizar. Si None, optimiza el mejor modelo
            de train_models(). Opciones: "logistic_regression",
            "svm", "random_forest", "xgboost", "adaboost".

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
            "svm": SVC(probability=True),
            "random_forest": RandomForestClassifier(random_state=42),
            "xgboost": XGBClassifier(
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            ),
            "adaboost": AdaBoostClassifier(random_state=42),
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
            "adaboost": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.5, 1.0],
                "model__algorithm": ["SAMME", "SAMME.R"],
            },
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimators[model_name]),
        ])

        # ── 3. GridSearchCV con TimeSeriesSplit ────────────
        # Pre-filtrar splits para excluir folds con una sola clase en training
        tscv_grid = TimeSeriesSplit(n_splits=ML_CV_SPLITS)
        cv_splits = [
            (tr, te) for tr, te in tscv_grid.split(self._X_train)
            if len(np.unique(self._y_train[tr])) >= 2
        ]

        if len(cv_splits) == 0:
            raise RuntimeError(
                "❌ Todos los folds de CV tienen una sola clase en training. "
                "El target es demasiado desbalanceado para optimizar. "
                "Intenta con más datos (last_n mayor) o una estrategia diferente."
            )

        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=cv_splits,
            scoring="f1",
            n_jobs=-1,
        )
        grid_search.fit(self._X_train, self._y_train)

        # ── 4. Actualizar modelo y métricas ────────────────
        self.model = grid_search.best_estimator_

        # Calcular nuevas métricas con cross-validation
        tscv = TimeSeriesSplit(n_splits=ML_CV_SPLITS)
        all_y_true = []
        all_y_pred = []

        for train_idx, test_idx in tscv.split(self._X_train):
            X_train = self._X_train[train_idx]
            X_test = self._X_train[test_idx]
            y_train = self._y_train[train_idx]
            y_test = self._y_train[test_idx]

            # Skip fold si solo hay una clase en training
            if len(np.unique(y_train)) < 2:
                continue

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        # Si todos los folds fueron skipped, métricas = 0
        if len(all_y_true) == 0:
            new_metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        else:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            new_metrics = {
                "accuracy": accuracy_score(all_y_true, all_y_pred),
                "precision": precision_score(all_y_true, all_y_pred, zero_division=0),
                "recall": recall_score(all_y_true, all_y_pred, zero_division=0),
                "f1": f1_score(all_y_true, all_y_pred, zero_division=0),
            }

        # Refit en datos de training
        self.model.fit(self._X_train, self._y_train)
        self.model_metrics = new_metrics

        # ── 5. Re-evaluación OOS ─────────────────────────
        old_oos = self._oos_metrics.copy() if self._oos_metrics else None

        if self._X_test is not None and len(self._X_test) > 0:
            y_pred_oos = self.model.predict(self._X_test)
            self._oos_metrics = {
                "accuracy": accuracy_score(self._y_test, y_pred_oos),
                "precision": precision_score(self._y_test, y_pred_oos, zero_division=0),
                "recall": recall_score(self._y_test, y_pred_oos, zero_division=0),
                "f1": f1_score(self._y_test, y_pred_oos, zero_division=0),
            }

        # ── 6. Print resultados ────────────────────────────
        print("=" * 85)
        print(f"🔧 OPTIMIZACIÓN — {model_name}")
        print("=" * 85)
        print(f"\nMejores parámetros: {grid_search.best_params_}")
        print(f"\n{'Métrica':<15} {'Antes':>10} {'Después':>10} {'Cambio':>10}")
        print("-" * 48)

        for metric in ["accuracy", "precision", "recall", "f1"]:
            old_val = old_metrics[metric]
            new_val = new_metrics[metric]
            diff = new_val - old_val
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"{metric:<15} {old_val:>10.4f} {new_val:>10.4f} {arrow} {abs(diff):>8.4f}")

        # Print OOS si disponible
        if self._oos_metrics is not None:
            print(f"\n{'─' * 50}")
            print(f"📊 Out-of-Sample (test set)")
            print(f"{'─' * 50}")

            if old_oos is not None:
                print(f"{'Métrica':<15} {'Antes':>10} {'Después':>10} {'Cambio':>10}")
                print("-" * 48)
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    ov = old_oos[metric]
                    nv = self._oos_metrics[metric]
                    diff = nv - ov
                    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                    print(f"{metric:<15} {ov:>10.4f} {nv:>10.4f} {arrow} {abs(diff):>8.4f}")
            else:
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    print(f"  {metric:<15} {self._oos_metrics[metric]:.4f}")

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
