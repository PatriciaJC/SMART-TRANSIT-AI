"""
main.py — API REST del Sistema Inteligente de Transporte Público.

Integra el simulador de datos, el modelo LSTM y el motor de decisiones
en una API completa y documentada con FastAPI.

Endpoints:
    GET  /                  → Estado del sistema
    GET  /health            → Health check
    GET  /simular           → Ejecuta la simulación y retorna datos
    GET  /predecir          → Predicciones del modelo LSTM
    POST /predecir          → Predicción con datos personalizados
    GET  /decidir           → Análisis con datos del dataset
    POST /decidir           → Análisis con datos personalizados
    GET  /entrenar          → Inicia el entrenamiento del modelo
    GET  /docs              → Documentación interactiva (Swagger)
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Asegurar que el directorio backend esté en el path
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    ensure_directories,
    logger,
    respuesta_ok,
    respuesta_error,
    timestamp_actual,
    DATASET_PATH,
    MODEL_PATH,
)
from simulador import ejecutar_simulacion
from modelo_lstm import predecir_desde_dataset, entrenar_modelo_completo
from decision_engine import analizar_estado


# ─────────────────────────────────────────────
# CONFIGURACIÓN DE LA APLICACIÓN
# ─────────────────────────────────────────────

app = FastAPI(
    title="Sistema Inteligente de Transporte Público",
    description=(
        "API backend para optimización de transporte urbano.\n\n"
        "Integra simulación de datos, predicciones LSTM y motor de decisiones "
        "basado en reglas para una gestión inteligente de la flota."
    ),
    version="1.0.0",
    contact={
        "name": "Equipo de Ingeniería",
        "email": "ingenieria@transporte-inteligente.com",
    },
    license_info={"name": "MIT"},
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permite conexión desde cualquier frontend (ajustar en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# MODELOS PYDANTIC (esquemas de entrada/salida)
# ─────────────────────────────────────────────

class PrediccionInput(BaseModel):
    """Datos de entrada para predicción personalizada."""
    hora: int = Field(default=8, ge=0, le=23, description="Hora del día (0-23)")
    es_hora_pico: int = Field(default=1, ge=0, le=1, description="1 si es hora pico")
    pasajeros: int = Field(default=45, ge=0, le=60, description="Número de pasajeros")
    densidad_trafico: float = Field(default=0.7, ge=0.0, le=1.0, description="Densidad [0-1]")
    velocidad_kmh: float = Field(default=30.0, ge=5.0, le=80.0, description="Velocidad km/h")
    ocupacion_pct: float = Field(default=75.0, ge=0.0, le=100.0, description="Ocupación %")
    indice_confiabilidad: float = Field(default=0.8, ge=0.0, le=1.0)
    frecuencia_optima_min: float = Field(default=8.0, ge=3.0, le=30.0)


class DecisionInput(BaseModel):
    """Datos de entrada para el motor de decisiones."""
    pasajeros: int = Field(default=45, ge=0, le=120, description="Pasajeros en el bus")
    capacidad_base: int = Field(default=60, ge=20, le=120)
    densidad_trafico: float = Field(default=0.7, ge=0.0, le=1.0)
    tiempo_estimado_min: float = Field(default=18.0, ge=1.0, le=120.0)
    tiempo_real_min: float = Field(default=22.0, ge=1.0, le=180.0)
    hora: int = Field(default=8, ge=0, le=23)
    es_hora_pico: bool = Field(default=True)
    incidente: bool = Field(default=False)
    tipo_incidente: str = Field(default="ninguno")
    ruta: str = Field(default="Ruta-1")
    clima: str = Field(default="despejado")
    capacidad_disponible: Optional[float] = Field(default=None)
    indice_confiabilidad: Optional[float] = Field(default=None)
    frecuencia_optima_min: Optional[float] = Field(default=None)


class SimulacionInput(BaseModel):
    """Parámetros opcionales para la simulación."""
    total_registros: int = Field(default=1200, ge=100, le=10000)
    guardar: bool = Field(default=True)


# ─────────────────────────────────────────────
# EVENTOS DE CICLO DE VIDA
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Inicialización al arrancar el servidor."""
    ensure_directories()
    logger.info("=" * 55)
    logger.info(" Sistema Inteligente de Transporte — INICIANDO")
    logger.info("=" * 55)
    logger.info(f"Dataset disponible : {os.path.exists(DATASET_PATH)}")
    logger.info(f"Modelo disponible  : {os.path.exists(MODEL_PATH)}")
    logger.info("Servidor listo para recibir solicitudes")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Sistema de transporte — Servidor detenido")


# ─────────────────────────────────────────────
# ENDPOINTS — SISTEMA
# ─────────────────────────────────────────────

@app.get("/", tags=["Sistema"], summary="Información del sistema")
async def raiz() -> Dict:
    """Retorna información general del sistema y estado de componentes."""
    return respuesta_ok(
        data={
            "sistema": "Sistema Inteligente de Transporte Público",
            "version": "1.0.0",
            "componentes": {
                "simulador": "activo",
                "modelo_lstm": "disponible" if os.path.exists(MODEL_PATH) else "no entrenado",
                "motor_decisiones": "activo",
                "dataset": "disponible" if os.path.exists(DATASET_PATH) else "no generado",
            },
            "endpoints": {
                "GET /simular": "Genera datos de simulación",
                "GET /predecir": "Predicciones del modelo LSTM",
                "POST /predecir": "Predicción con datos personalizados",
                "GET /decidir": "Análisis del motor de decisiones",
                "POST /decidir": "Decisión con datos personalizados",
                "GET /entrenar": "Entrenar el modelo LSTM",
                "GET /docs": "Documentación Swagger interactiva",
            },
        },
        mensaje="Sistema en operación",
    )


@app.get("/health", tags=["Sistema"], summary="Health check")
async def health_check() -> Dict:
    """Endpoint de salud para monitoreo y balanceadores de carga."""
    return {
        "status": "healthy",
        "timestamp": timestamp_actual(),
        "dataset_ok": os.path.exists(DATASET_PATH),
        "modelo_ok": os.path.exists(MODEL_PATH),
    }


# ─────────────────────────────────────────────
# ENDPOINTS — SIMULADOR
# ─────────────────────────────────────────────

@app.get(
    "/simular",
    tags=["Simulador"],
    summary="Ejecutar simulación de datos",
    response_description="Dataset simulado con resumen estadístico y muestra",
)
async def get_simular(
    registros: int = Query(
        default=1200,
        ge=100,
        le=10000,
        description="Cantidad de registros a generar",
    ),
    guardar: bool = Query(
        default=True,
        description="Si True, guarda el CSV en /data/dataset.csv",
    ),
) -> Dict:
    """
    Ejecuta el simulador de datos de transporte público.

    Genera registros sintéticos con variables reales:
    pasajeros, tráfico, tiempos, confiabilidad y métricas derivadas.
    """
    try:
        logger.info(f"Solicitud de simulación: {registros} registros")
        resultado = ejecutar_simulacion(
            total_registros=registros,
            guardar=guardar,
        )
        return respuesta_ok(
            data=resultado,
            mensaje=f"Simulación completada: {resultado['resumen']['total_registros']} registros generados",
        )
    except Exception as e:
        logger.error(f"Error en simulación: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/simular",
    tags=["Simulador"],
    summary="Simulación con parámetros personalizados",
)
async def post_simular(params: SimulacionInput) -> Dict:
    """Ejecuta la simulación con parámetros específicos en el body."""
    try:
        resultado = ejecutar_simulacion(
            total_registros=params.total_registros,
            guardar=params.guardar,
        )
        return respuesta_ok(data=resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# ENDPOINTS — MODELO LSTM
# ─────────────────────────────────────────────

@app.get(
    "/predecir",
    tags=["Modelo LSTM"],
    summary="Predicciones del modelo LSTM",
    response_description="Lista de predicciones vs valores reales",
)
async def get_predecir(
    muestras: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Número de predicciones a generar",
    )
) -> Dict:
    """
    Usa el modelo LSTM entrenado para predecir tiempos de recorrido
    sobre las últimas muestras del dataset.

    **Requisitos**: El modelo debe estar entrenado (`GET /entrenar`).
    """
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Modelo LSTM no encontrado. "
                    "Ejecuta primero GET /entrenar para entrenarlo."
                ),
            )

        logger.info(f"Solicitud de predicción: {muestras} muestras")
        predicciones = predecir_desde_dataset(n_muestras=muestras)

        if predicciones and "error" in predicciones[0]:
            raise HTTPException(status_code=500, detail=predicciones[0]["error"])

        return respuesta_ok(
            data={
                "predicciones": predicciones,
                "total": len(predicciones),
                "modelo": MODEL_PATH,
            },
            mensaje=f"Predicciones generadas: {len(predicciones)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predecir",
    tags=["Modelo LSTM"],
    summary="Predicción con datos personalizados",
)
async def post_predecir(datos: PrediccionInput) -> Dict:
    """
    Realiza una predicción individual con datos proporcionados en el body.

    Útil para integración con frontend en tiempo real.
    """
    try:
        import numpy as np
        from modelo_lstm import ModeloLSTMTransporte, LONGITUD_SECUENCIA, CARACTERISTICAS

        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=404,
                detail="Modelo no entrenado. Ejecuta GET /entrenar primero.",
            )

        modelo = ModeloLSTMTransporte()
        if not modelo.cargar_modelo_guardado():
            raise HTTPException(status_code=500, detail="Error al cargar el modelo")

        # Construir secuencia repitiendo el vector de entrada (simulación de ventana)
        vector = np.array(
            [
                datos.hora,
                datos.es_hora_pico,
                datos.pasajeros,
                datos.densidad_trafico,
                datos.velocidad_kmh,
                datos.ocupacion_pct,
                datos.indice_confiabilidad,
                datos.frecuencia_optima_min,
            ],
            dtype=np.float32,
        )

        # Normalizar usando el scaler cargado
        n_features_total = len(CARACTERISTICAS) + 1
        dummy_full = np.zeros((1, n_features_total), dtype=np.float32)
        dummy_full[0, : len(CARACTERISTICAS)] = vector
        vector_norm = modelo.scaler.transform(dummy_full)[0, : len(CARACTERISTICAS)]

        # Crear secuencia repitiendo el vector normalizado
        secuencia = np.tile(vector_norm, (LONGITUD_SECUENCIA, 1))
        secuencia = secuencia.reshape(1, LONGITUD_SECUENCIA, len(CARACTERISTICAS))

        tiempo_predicho = modelo.predecir(secuencia)

        return respuesta_ok(
            data={
                "tiempo_predicho_min": tiempo_predicho,
                "datos_entrada": datos.model_dump(),
            },
            mensaje="Predicción completada",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción personalizada: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# ENDPOINTS — MOTOR DE DECISIONES
# ─────────────────────────────────────────────

@app.get(
    "/decidir",
    tags=["Motor de Decisiones"],
    summary="Análisis con datos del dataset",
    response_description="Acciones y alertas recomendadas",
)
async def get_decidir(
    pasajeros: int = Query(default=50, ge=0, le=120),
    densidad_trafico: float = Query(default=0.75, ge=0.0, le=1.0),
    tiempo_estimado_min: float = Query(default=18.0, ge=1.0),
    tiempo_real_min: float = Query(default=25.0, ge=1.0),
    hora: int = Query(default=8, ge=0, le=23),
    es_hora_pico: bool = Query(default=True),
    incidente: bool = Query(default=False),
    tipo_incidente: str = Query(default="ninguno"),
    ruta: str = Query(default="Ruta-1"),
    clima: str = Query(default="despejado"),
) -> Dict:
    """
    Analiza el estado actual del sistema y devuelve decisiones operativas.

    El motor evalúa:
    - Estado de capacidad del bus
    - Puntualidad e índice de confiabilidad
    - Nivel de demanda y frecuencia óptima
    - Condiciones de tráfico
    - Incidentes activos
    - Hora pico
    """
    try:
        datos = {
            "pasajeros": pasajeros,
            "densidad_trafico": densidad_trafico,
            "tiempo_estimado_min": tiempo_estimado_min,
            "tiempo_real_min": tiempo_real_min,
            "hora": hora,
            "es_hora_pico": es_hora_pico,
            "incidente": incidente,
            "tipo_incidente": tipo_incidente,
            "ruta": ruta,
            "clima": clima,
        }

        logger.info(f"Solicitud de decisión — Ruta: {ruta}, Hora: {hora}:00")
        resultado = analizar_estado(datos)

        if "error" in resultado:
            raise HTTPException(status_code=422, detail=resultado["error"])

        return respuesta_ok(
            data=resultado,
            mensaje=f"Análisis completado: {resultado['total_acciones']} acciones generadas",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en motor de decisiones: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/decidir",
    tags=["Motor de Decisiones"],
    summary="Decisión con datos personalizados (body)",
)
async def post_decidir(datos: DecisionInput) -> Dict:
    """
    Recibe el estado del sistema como JSON y retorna acciones priorizadas.

    Ideal para integración con frontend en tiempo real o sistema SCADA.
    """
    try:
        resultado = analizar_estado(datos.model_dump())

        if "error" in resultado:
            raise HTTPException(status_code=422, detail=resultado["error"])

        return respuesta_ok(data=resultado, mensaje="Decisión generada")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# ENDPOINTS — ENTRENAMIENTO
# ─────────────────────────────────────────────

@app.get(
    "/entrenar",
    tags=["Modelo LSTM"],
    summary="Entrenar el modelo LSTM",
    response_description="Métricas de entrenamiento y rutas de artefactos",
)
async def get_entrenar() -> Dict:
    """
    Ejecuta el pipeline completo de entrenamiento del modelo LSTM.

    Pasos:
    1. Carga el dataset (debes ejecutar `/simular` primero)
    2. Preprocesa y normaliza los datos
    3. Crea secuencias temporales de longitud 24
    4. Divide en train/test (80/20) respetando el orden temporal
    5. Entrena la red LSTM con early stopping
    6. Evalúa con MAE, RMSE y MAPE
    7. Guarda el modelo en `/models/modelo_lstm.keras`

    ⚠️ Este proceso puede tardar varios minutos según el hardware.
    """
    try:
        if not os.path.exists(DATASET_PATH):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Dataset no encontrado. "
                    "Ejecuta primero GET /simular para generar los datos."
                ),
            )

        logger.info("Iniciando entrenamiento del modelo LSTM...")
        resultado = entrenar_modelo_completo()

        return respuesta_ok(
            data=resultado,
            mensaje="Modelo entrenado y guardado correctamente",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el entrenamiento: {str(e)}",
        )


# ─────────────────────────────────────────────
# ENDPOINT — ANÁLISIS INTEGRADO
# ─────────────────────────────────────────────

@app.get(
    "/analisis-completo",
    tags=["Sistema"],
    summary="Análisis integrado: simulación + predicción + decisión",
)
async def analisis_completo(
    pasajeros: int = Query(default=50),
    hora: int = Query(default=8, ge=0, le=23),
    densidad_trafico: float = Query(default=0.75, ge=0.0, le=1.0),
    ruta: str = Query(default="Ruta-1"),
) -> Dict:
    """
    Endpoint integrador que combina los tres componentes del sistema:

    1. **Simulador**: genera un snapshot del estado actual
    2. **Motor de decisiones**: analiza el estado y emite acciones
    3. **Modelo LSTM**: predice tiempo de recorrido (si está entrenado)

    Ideal para el dashboard del frontend.
    """
    from utils import (
        calcular_capacidad_dinamica,
        calcular_frecuencia_optima,
        calcular_indice_confiabilidad,
        calcular_velocidad_trafico,
    )

    try:
        # 1. Calcular métricas del estado actual
        velocidad = calcular_velocidad_trafico(50.0, densidad_trafico)
        tiempo_estimado = round((12.0 / velocidad) * 60.0, 2)
        factor_hora = 1.0 + (0.3 if hora in range(7, 10) or hora in range(17, 20) else 0.0)
        tiempo_real = round(tiempo_estimado * factor_hora, 2)
        capacidad_disponible = calcular_capacidad_dinamica(pasajeros)
        i_cs = calcular_indice_confiabilidad(tiempo_estimado, tiempo_real)
        frecuencia = calcular_frecuencia_optima(pasajeros)
        es_hora_pico = hora in range(7, 10) or hora in range(17, 20)

        estado_actual = {
            "ruta": ruta,
            "hora": hora,
            "pasajeros": pasajeros,
            "densidad_trafico": densidad_trafico,
            "velocidad_kmh": velocidad,
            "tiempo_estimado_min": tiempo_estimado,
            "tiempo_real_min": tiempo_real,
            "capacidad_disponible": capacidad_disponible,
            "indice_confiabilidad": i_cs,
            "frecuencia_optima_min": frecuencia,
            "es_hora_pico": es_hora_pico,
            "ocupacion_pct": round(pasajeros / 60 * 100, 2),
        }

        # 2. Motor de decisiones
        decision = analizar_estado({
            **estado_actual,
            "capacidad_base": 60,
            "incidente": False,
            "tipo_incidente": "ninguno",
            "clima": "despejado",
        })

        # 3. Predicción LSTM (si disponible)
        prediccion_lstm = None
        if os.path.exists(MODEL_PATH):
            try:
                preds = predecir_desde_dataset(n_muestras=1)
                if preds and "error" not in preds[0]:
                    prediccion_lstm = preds[0]
            except Exception:
                prediccion_lstm = {"mensaje": "Predicción no disponible"}

        return respuesta_ok(
            data={
                "estado_actual": estado_actual,
                "decision": decision,
                "prediccion_lstm": prediccion_lstm,
                "timestamp": timestamp_actual(),
            },
            mensaje="Análisis integrado completado",
        )
    except Exception as e:
        logger.error(f"Error en análisis completo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # Hot-reload en desarrollo
        log_level="info",
    )


# ─────────────────────────────────────────────
# ENDPOINTS — RUTAS REALES MIBUS
# (añadidos para coherencia con routes.csv / route_stops.csv)
# ─────────────────────────────────────────────

from utils import RUTAS_REALES, RUTAS_ACTIVAS, PARADAS_POR_RUTA, get_capacidad, get_distancia


@app.get("/rutas", tags=["Rutas MiBus"], summary="Listado de rutas reales MiBus")
async def get_rutas(
    solo_activas: bool = Query(default=True, description="Si True, solo rutas en estado 'Completas'"),
    familia:      Optional[str] = Query(default=None, description="Filtrar por familia: E, C, S, T, M"),
) -> Dict:
    """
    Retorna el catálogo completo de rutas MiBus con capacidad real por familia.

    Familias disponibles:
    - **E** — Eléctricos articulados (80 asientos)
    - **C** — Convencionales circulares (70 asientos)
    - **S** — Corredor Sur (65 asientos)
    - **T** — Corredor Norte (65 asientos)
    - **M** — Metropolitanos medianos (55 asientos)
    """
    rutas = {
        k: {**v, "capacidad": get_capacidad(k), "distancia_km": get_distancia(k)}
        for k, v in RUTAS_REALES.items()
        if (not solo_activas or v["estado"] == "Completas")
        and (familia is None or v["familia"] == familia.upper())
    }
    return respuesta_ok(
        data={
            "rutas":    rutas,
            "total":    len(rutas),
            "familias": sorted(set(v["familia"] for v in rutas.values())),
        },
        mensaje=f"{len(rutas)} rutas encontradas",
    )


@app.get("/rutas/{numero}", tags=["Rutas MiBus"], summary="Detalle de una ruta")
async def get_ruta_detalle(numero: str) -> Dict:
    """Retorna información completa de una ruta específica, incluyendo paradas."""
    numero = numero.upper()
    if numero not in RUTAS_REALES:
        raise HTTPException(
            status_code=404,
            detail=f"Ruta '{numero}' no encontrada. Rutas disponibles: {list(RUTAS_REALES.keys())}",
        )
    info = RUTAS_REALES[numero]
    return respuesta_ok(
        data={
            "numero":        numero,
            "nombre":        info["nombre"],
            "familia":       info["familia"],
            "estado":        info["estado"],
            "terminal_inicio": info["terminal_inicio"],
            "terminal_fin":    info["terminal_fin"],
            "capacidad":     get_capacidad(numero),
            "distancia_km":  get_distancia(numero),
            "paradas":       PARADAS_POR_RUTA.get(numero, []),
            "tiene_paradas": numero in PARADAS_POR_RUTA,
        }
    )


@app.get("/rutas/{numero}/paradas", tags=["Rutas MiBus"], summary="Paradas de una ruta")
async def get_paradas_ruta(numero: str) -> Dict:
    """Retorna la lista ordenada de paradas de una ruta específica."""
    numero = numero.upper()
    if numero not in RUTAS_REALES:
        raise HTTPException(status_code=404, detail=f"Ruta '{numero}' no encontrada")
    return respuesta_ok(
        data={
            "ruta":    numero,
            "nombre":  RUTAS_REALES[numero]["nombre"],
            "paradas": PARADAS_POR_RUTA.get(numero, []),
            "total_paradas": len(PARADAS_POR_RUTA.get(numero, [])),
        }
    )