"""
modelo_lstm.py — Modelo LSTM para predicción de tiempos de recorrido.

Carga el dataset, preprocesa datos, crea secuencias temporales,
entrena una red LSTM y guarda el modelo y el scaler para inferencia.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    DATASET_PATH,
    MODEL_PATH,
    SCALER_PATH,
    ensure_directories,
    logger,
)

# ─────────────────────────────────────────────
# HIPERPARÁMETROS DEL MODELO
# ─────────────────────────────────────────────

LONGITUD_SECUENCIA = 24        # Ventana temporal (24 horas)
CARACTERISTICAS = [            # Variables de entrada (features)
    "hora",
    "es_hora_pico",
    "pasajeros",
    "densidad_trafico",
    "velocidad_kmh",
    "ocupacion_pct",
    "indice_confiabilidad",
    "frecuencia_optima_min",
]
VARIABLE_OBJETIVO = "tiempo_real_min"   # Variable a predecir
PROPORCION_TEST = 0.20                  # 20% para evaluación
EPOCHS = 40
BATCH_SIZE = 32
UNIDADES_LSTM_1 = 64
UNIDADES_LSTM_2 = 32
DROPOUT_RATE = 0.20
LEARNING_RATE = 0.001


# ─────────────────────────────────────────────
# CLASE PRINCIPAL DEL MODELO
# ─────────────────────────────────────────────

class ModeloLSTMTransporte:
    """
    Red neuronal LSTM para predicción de tiempos de recorrido
    en el sistema de transporte público.

    Flujo:
        cargar_datos() → preprocesar() → crear_secuencias()
        → construir_modelo() → entrenar() → evaluar() → guardar()
    """

    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.historial = None
        self.mae_test: Optional[float] = None
        self.rmse_test: Optional[float] = None
        self._modelo_listo = False
        logger.info("ModeloLSTMTransporte inicializado")

    # ─────────────────────────────────────────
    # 1. CARGA DE DATOS
    # ─────────────────────────────────────────

    def cargar_datos(self, ruta: str = DATASET_PATH) -> pd.DataFrame:
        """Carga el dataset desde CSV y valida columnas requeridas."""
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"Dataset no encontrado en: {ruta}\n"
                "Ejecuta primero: python simulador.py"
            )

        df = pd.read_csv(ruta, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        columnas_requeridas = CARACTERISTICAS + [VARIABLE_OBJETIVO]
        faltantes = [c for c in columnas_requeridas if c not in df.columns]
        if faltantes:
            raise ValueError(f"Columnas faltantes en el dataset: {faltantes}")

        logger.info(f"Dataset cargado: {len(df)} registros — {len(df.columns)} columnas")
        return df

    # ─────────────────────────────────────────
    # 2. PREPROCESAMIENTO
    # ─────────────────────────────────────────

    def preprocesar(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normaliza features y objetivo con MinMaxScaler.

        Retorna:
            X — array normalizado de características (n, num_features+1)
            scaler — scaler ajustado (se guarda en self.scaler)
        """
        from sklearn.preprocessing import MinMaxScaler

        columnas_modelo = CARACTERISTICAS + [VARIABLE_OBJETIVO]
        datos = df[columnas_modelo].values.astype(np.float32)

        # Un solo scaler para todas las columnas (features + objetivo)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        datos_normalizados = self.scaler.fit_transform(datos)

        logger.info(
            f"Preprocesamiento completado — Shape: {datos_normalizados.shape}"
        )
        return datos_normalizados

    # ─────────────────────────────────────────
    # 3. CREACIÓN DE SECUENCIAS
    # ─────────────────────────────────────────

    def crear_secuencias(
        self, datos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforma el array 2D en secuencias (X, y) para el LSTM.

        X shape: (n_muestras, LONGITUD_SECUENCIA, n_features)
        y shape: (n_muestras,)

        El objetivo (última columna) es el índice del tiempo real.
        """
        X, y = [], []
        n_features = len(CARACTERISTICAS)
        idx_objetivo = -1  # Última columna = tiempo_real_min normalizado

        for i in range(LONGITUD_SECUENCIA, len(datos)):
            # Secuencia de features (todas menos el objetivo)
            X.append(datos[i - LONGITUD_SECUENCIA : i, :n_features])
            # Valor objetivo del instante siguiente
            y.append(datos[i, idx_objetivo])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(f"Secuencias creadas — X: {X.shape}, y: {y.shape}")
        return X, y

    # ─────────────────────────────────────────
    # 4. SPLIT ENTRENAMIENTO / TEST
    # ─────────────────────────────────────────

    def dividir_datos(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divide los datos respetando el orden temporal (sin shuffle)."""
        n_test = int(len(X) * PROPORCION_TEST)
        n_train = len(X) - n_test

        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        logger.info(
            f"Split temporal — Entrenamiento: {n_train}, Test: {n_test}"
        )
        return X_train, X_test, y_train, y_test

    # ─────────────────────────────────────────
    # 5. CONSTRUCCIÓN DEL MODELO
    # ─────────────────────────────────────────

    def construir_modelo(self, n_features: int):
        """
        Arquitectura LSTM de dos capas con regularización Dropout.

        Input  → LSTM(64, return_sequences=True) → Dropout(0.2)
               → LSTM(32) → Dropout(0.2)
               → Dense(16, relu) → Dense(1)
        """
        import tensorflow as tf
        from tensorflow.keras.layers import (
            LSTM, Dense, Dropout, Input
        )
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam

        self.modelo = Sequential(
            [
                Input(shape=(LONGITUD_SECUENCIA, n_features)),
                LSTM(
                    UNIDADES_LSTM_1,
                    return_sequences=True,
                    name="lstm_capa_1",
                ),
                Dropout(DROPOUT_RATE, name="dropout_1"),
                LSTM(
                    UNIDADES_LSTM_2,
                    return_sequences=False,
                    name="lstm_capa_2",
                ),
                Dropout(DROPOUT_RATE, name="dropout_2"),
                Dense(16, activation="relu", name="dense_1"),
                Dense(1, name="salida"),
            ],
            name="modelo_lstm_transporte",
        )

        self.modelo.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="mean_squared_error",
            metrics=["mae"],
        )

        logger.info("Modelo construido:")
        self.modelo.summary(print_fn=logger.info)
        return self.modelo

    # ─────────────────────────────────────────
    # 6. ENTRENAMIENTO
    # ─────────────────────────────────────────

    def entrenar(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """Entrena el modelo con early stopping y reduce LR on plateau."""
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        logger.info(f"Iniciando entrenamiento — {EPOCHS} épocas máximo")
        self.historial = self.modelo.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("Entrenamiento completado")
        return self.historial

    # ─────────────────────────────────────────
    # 7. EVALUACIÓN
    # ─────────────────────────────────────────

    def evaluar(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Evalúa el modelo en el conjunto de test y calcula métricas
        desnormalizadas (en minutos reales).
        """
        predicciones_norm = self.modelo.predict(X_test, verbose=0).flatten()

        # Desnormalizar: reconstruir el array completo para inverse_transform
        n_features_total = len(CARACTERISTICAS) + 1  # +1 = objetivo
        dummy = np.zeros((len(predicciones_norm), n_features_total), dtype=np.float32)
        dummy[:, -1] = predicciones_norm
        predicciones_min = self.scaler.inverse_transform(dummy)[:, -1]

        dummy_real = np.zeros((len(y_test), n_features_total), dtype=np.float32)
        dummy_real[:, -1] = y_test
        reales_min = self.scaler.inverse_transform(dummy_real)[:, -1]

        self.mae_test = float(np.mean(np.abs(predicciones_min - reales_min)))
        self.rmse_test = float(np.sqrt(np.mean((predicciones_min - reales_min) ** 2)))

        metricas = {
            "mae_minutos": round(self.mae_test, 4),
            "rmse_minutos": round(self.rmse_test, 4),
            "mape_pct": round(
                float(
                    np.mean(
                        np.abs((reales_min - predicciones_min) / np.maximum(reales_min, 1e-6))
                    )
                )
                * 100,
                2,
            ),
            "muestras_test": len(y_test),
        }

        logger.info(
            f"Evaluación — MAE: {metricas['mae_minutos']} min | "
            f"RMSE: {metricas['rmse_minutos']} min | "
            f"MAPE: {metricas['mape_pct']}%"
        )
        self._modelo_listo = True
        return metricas

    # ─────────────────────────────────────────
    # 8. PERSISTENCIA
    # ─────────────────────────────────────────

    def guardar(self) -> Dict[str, str]:
        """Guarda el modelo Keras y el scaler en disco."""
        ensure_directories()

        self.modelo.save(MODEL_PATH)
        logger.info(f"Modelo guardado en: {MODEL_PATH}")

        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler guardado en: {SCALER_PATH}")

        return {"modelo": MODEL_PATH, "scaler": SCALER_PATH}

    def cargar_modelo_guardado(self) -> bool:
        """Carga modelo y scaler desde disco para inferencia."""
        import tensorflow as tf

        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            logger.warning("Modelo o scaler no encontrados. Entrena primero el modelo.")
            return False

        self.modelo = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        self._modelo_listo = True
        logger.info("Modelo y scaler cargados correctamente")
        return True

    # ─────────────────────────────────────────
    # 9. PREDICCIÓN EN TIEMPO REAL
    # ─────────────────────────────────────────

    def predecir(self, secuencia: np.ndarray) -> float:
        """
        Realiza una predicción sobre una secuencia de entrada.

        Parámetros:
            secuencia — array shape (1, LONGITUD_SECUENCIA, n_features)

        Retorna:
            float — tiempo predicho en minutos (desnormalizado)
        """
        if not self._modelo_listo:
            raise RuntimeError("El modelo no está listo. Entrena o carga un modelo primero.")

        prediccion_norm = self.modelo.predict(secuencia, verbose=0).flatten()[0]

        n_features_total = len(CARACTERISTICAS) + 1
        dummy = np.zeros((1, n_features_total), dtype=np.float32)
        dummy[0, -1] = prediccion_norm
        tiempo_min = float(self.scaler.inverse_transform(dummy)[0, -1])

        return round(max(0.0, tiempo_min), 2)


# ─────────────────────────────────────────────
# PIPELINE COMPLETO PARA LA API
# ─────────────────────────────────────────────

def entrenar_modelo_completo() -> Dict:
    """
    Ejecuta el pipeline completo de entrenamiento:
    carga → preprocesa → secuencias → construye → entrena → evalúa → guarda

    Retorna un dict con métricas y rutas de los artefactos guardados.
    """
    modelo = ModeloLSTMTransporte()

    df = modelo.cargar_datos()
    datos_norm = modelo.preprocesar(df)
    X, y = modelo.crear_secuencias(datos_norm)
    X_train, X_test, y_train, y_test = modelo.dividir_datos(X, y)
    modelo.construir_modelo(n_features=len(CARACTERISTICAS))
    modelo.entrenar(X_train, y_train, X_test, y_test)
    metricas = modelo.evaluar(X_test, y_test)
    rutas = modelo.guardar()

    return {
        "estado": "entrenado",
        "metricas": metricas,
        "artefactos": rutas,
        "arquitectura": {
            "tipo": "LSTM",
            "capas": [
                f"LSTM({UNIDADES_LSTM_1}, return_sequences=True)",
                f"Dropout({DROPOUT_RATE})",
                f"LSTM({UNIDADES_LSTM_2})",
                f"Dropout({DROPOUT_RATE})",
                "Dense(16, relu)",
                "Dense(1)",
            ],
            "longitud_secuencia": LONGITUD_SECUENCIA,
            "features": CARACTERISTICAS,
            "objetivo": VARIABLE_OBJETIVO,
        },
    }


def predecir_desde_dataset(n_muestras: int = 5) -> List[Dict]:
    """
    Carga el modelo y realiza predicciones sobre las últimas n_muestras del dataset.
    Útil para el endpoint GET /predecir de la API.
    """
    modelo = ModeloLSTMTransporte()

    if not modelo.cargar_modelo_guardado():
        return [{"error": "Modelo no encontrado. Ejecuta el entrenamiento primero."}]

    df = modelo.cargar_datos()
    datos_norm = modelo.preprocesar(df)

    resultados = []
    # Tomar las últimas n_muestras secuencias disponibles
    inicio = len(datos_norm) - n_muestras - LONGITUD_SECUENCIA
    for i in range(n_muestras):
        idx = inicio + i
        if idx < 0:
            continue
        secuencia = datos_norm[idx : idx + LONGITUD_SECUENCIA, : len(CARACTERISTICAS)]
        secuencia = secuencia.reshape(1, LONGITUD_SECUENCIA, len(CARACTERISTICAS))
        tiempo_pred = modelo.predecir(secuencia)
        tiempo_real = df[VARIABLE_OBJETIVO].iloc[idx + LONGITUD_SECUENCIA]

        resultados.append(
            {
                "muestra": i + 1,
                "timestamp": str(df["timestamp"].iloc[idx + LONGITUD_SECUENCIA]),
                "tiempo_real_min": round(float(tiempo_real), 2),
                "tiempo_predicho_min": tiempo_pred,
                "error_abs_min": round(abs(tiempo_real - tiempo_pred), 2),
            }
        )

    return resultados


# ─────────────────────────────────────────────
# EJECUCIÓN DIRECTA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 Iniciando entrenamiento del modelo LSTM...")
    resultado = entrenar_modelo_completo()

    print("\n✅ Entrenamiento completado")
    print(f"   MAE  : {resultado['metricas']['mae_minutos']} minutos")
    print(f"   RMSE : {resultado['metricas']['rmse_minutos']} minutos")
    print(f"   MAPE : {resultado['metricas']['mape_pct']}%")
    print(f"\n💾 Artefactos guardados:")
    for k, v in resultado["artefactos"].items():
        print(f"   {k}: {v}")