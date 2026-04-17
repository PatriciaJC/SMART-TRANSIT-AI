"""
simulador.py — Generador de datos sintéticos coherente con rutas reales MiBus.
Cada registro usa la capacidad y distancia correcta según la familia de la ruta.
"""

import os, random, logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    DATASET_PATH, RUTAS_ACTIVAS, RUTAS_REALES,
    get_capacidad, get_distancia,
    calcular_capacidad_dinamica, calcular_frecuencia_optima,
    calcular_indice_confiabilidad, calcular_velocidad_trafico,
    ensure_directories, logger,
)

TOTAL_REGISTROS   = 1200
VELOCIDAD_BASE    = 50.0   # km/h en vía libre

DIAS_SEMANA = ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"]
CLIMA_TIPOS = ["despejado","nublado","lluvia_leve","lluvia_intensa"]
CLIMA_IMPACTO = {"despejado":1.0,"nublado":0.95,"lluvia_leve":0.85,"lluvia_intensa":0.65}

PATRON_DEMANDA = {
    0:0.05,1:0.03,2:0.02,3:0.02,4:0.05,5:0.15,6:0.50,7:0.85,8:1.00,9:0.80,
    10:0.60,11:0.65,12:0.75,13:0.70,14:0.60,15:0.65,16:0.80,17:0.95,
    18:1.00,19:0.85,20:0.60,21:0.40,22:0.25,23:0.12,
}
PATRON_TRAFICO = {
    0:0.05,1:0.03,2:0.02,3:0.02,4:0.08,5:0.20,6:0.55,7:0.85,8:0.95,9:0.75,
    10:0.55,11:0.60,12:0.70,13:0.68,14:0.60,15:0.65,16:0.80,17:0.92,
    18:0.90,19:0.75,20:0.50,21:0.30,22:0.18,23:0.08,
}

def _es_pico(hora:int)->bool:
    return hora in range(7,10) or hora in range(17,20)

def _factor_dia(dia:str)->float:
    return {"lunes":1.0,"martes":0.98,"miércoles":0.95,"jueves":0.97,
            "viernes":1.05,"sábado":0.70,"domingo":0.45}.get(dia,1.0)

def _pasajeros(hora:int,dia:str,clima:str,capacidad:int)->int:
    base = PATRON_DEMANDA[hora]*capacidad*_factor_dia(dia)*CLIMA_IMPACTO[clima]
    ruido = np.random.normal(0,base*0.12)
    return int(round(max(0,min(capacidad,base+ruido))))

def _densidad(hora:int,dia:str,clima:str)->float:
    t = PATRON_TRAFICO[hora]
    if dia in ["sábado","domingo"]: t*=0.55
    if clima=="lluvia_intensa": t=min(1.0,t*1.30)
    elif clima=="lluvia_leve":  t=min(1.0,t*1.12)
    return round(max(0.0,min(1.0,t+np.random.normal(0,0.05))),4)

def _incidente()->Tuple[bool,Optional[str]]:
    tipos=["accidente","avería_mecánica","bloqueo_vial","manifestación"]
    if random.random()<0.05: return True,random.choice(tipos)
    return False,None


class SimuladorTransporte:
    def __init__(self,total=TOTAL_REGISTROS,semilla=42):
        self.total=total
        np.random.seed(semilla); random.seed(semilla)

    def generar_registro(self,fecha:datetime)->Dict:
        hora  = fecha.hour
        dia   = DIAS_SEMANA[fecha.weekday()]
        clima = random.choices(CLIMA_TIPOS,weights=[0.50,0.25,0.18,0.07])[0]

        # ── CAMBIO CLAVE: ruta y métricas reales ──
        ruta     = random.choice(RUTAS_ACTIVAS)
        cap      = get_capacidad(ruta)          # capacidad real por familia
        dist     = get_distancia(ruta)          # distancia real por familia
        familia  = RUTAS_REALES[ruta]["familia"]
        nombre   = RUTAS_REALES[ruta]["nombre"]
        t_inicio = RUTAS_REALES[ruta]["terminal_inicio"]
        t_fin    = RUTAS_REALES[ruta]["terminal_fin"]

        pasajeros    = _pasajeros(hora,dia,clima,cap)
        densidad     = _densidad(hora,dia,clima)
        velocidad    = calcular_velocidad_trafico(VELOCIDAD_BASE,densidad)
        t_estimado   = round((dist/velocidad)*60.0,2)

        # Tiempo real con variación contextual
        desv = 0.08
        if _es_pico(hora): desv+=0.10
        if clima=="lluvia_intensa": desv+=0.15
        elif clima=="lluvia_leve":  desv+=0.07
        t_real = round(max(t_estimado*0.7, t_estimado*np.random.normal(1.0,desv)),2)

        cap_disp = calcular_capacidad_dinamica(pasajeros,cap)
        i_cs     = calcular_indice_confiabilidad(t_estimado,t_real)
        freq_opt = calcular_frecuencia_optima(pasajeros,cap)
        incid,tipo_inc = _incidente()

        return {
            "timestamp":          fecha.strftime("%Y-%m-%d %H:%M:%S"),
            "hora":               hora,
            "minuto":             fecha.minute,
            "dia_semana":         dia,
            # ── campos reales de ruta ──
            "ruta":               ruta,
            "nombre_ruta":        nombre,
            "familia":            familia,
            "terminal_inicio":    t_inicio,
            "terminal_fin":       t_fin,
            "clima":              clima,
            # demanda
            "pasajeros":          pasajeros,
            "capacidad_base":     cap,           # varía según familia
            "capacidad_disponible": cap_disp,
            "ocupacion_pct":      round(pasajeros/cap*100,2),
            # tráfico
            "densidad_trafico":   densidad,
            "velocidad_kmh":      velocidad,
            # tiempos — basados en distancia real
            "distancia_km":       dist,
            "tiempo_estimado_min": t_estimado,
            "tiempo_real_min":     t_real,
            "desvio_tiempo_min":   round(t_real-t_estimado,2),
            # métricas
            "indice_confiabilidad":   i_cs,
            "frecuencia_optima_min":  freq_opt,
            "es_hora_pico":           int(_es_pico(hora)),
            "incidente":              int(incid),
            "tipo_incidente":         tipo_inc if incid else "ninguno",
        }

    def generar_dataset(self)->pd.DataFrame:
        logger.info("Generando dataset con rutas reales MiBus…")
        registros=[]
        fecha=datetime.now()-timedelta(days=50)
        for i in range(self.total):
            registros.append(self.generar_registro(fecha))
            fecha+=timedelta(hours=1)
            if (i+1)%200==0: logger.info(f"  {i+1}/{self.total}")
        df=pd.DataFrame(registros)
        df.sort_values("timestamp",inplace=True)
        df.reset_index(drop=True,inplace=True)
        return df

    def guardar_dataset(self,df:pd.DataFrame)->str:
        ensure_directories()
        df.to_csv(DATASET_PATH,index=False,encoding="utf-8")
        logger.info(f"Dataset guardado: {DATASET_PATH}")
        return DATASET_PATH

    def resumen_estadistico(self,df:pd.DataFrame)->Dict:
        return {
            "total_registros": len(df),
            "columnas":        list(df.columns),
            "rango_fechas":    {"inicio":df["timestamp"].iloc[0],"fin":df["timestamp"].iloc[-1]},
            "estadisticas": {
                "pasajeros_promedio":    round(df["pasajeros"].mean(),2),
                "ocupacion_promedio_pct":round(df["ocupacion_pct"].mean(),2),
                "velocidad_promedio_kmh":round(df["velocidad_kmh"].mean(),2),
                "confiabilidad_promedio":round(df["indice_confiabilidad"].mean(),4),
                "incidentes_total":      int(df["incidente"].sum()),
                "horas_pico_registros":  int(df["es_hora_pico"].sum()),
            },
            "distribucion_clima":   df["clima"].value_counts().to_dict(),
            "distribucion_rutas":   df["ruta"].value_counts().to_dict(),
            "distribucion_familia": df["familia"].value_counts().to_dict(),
        }


def ejecutar_simulacion(total_registros=TOTAL_REGISTROS,guardar=True)->Dict:
    sim=SimuladorTransporte(total_registros=total_registros)
    df=sim.generar_dataset()
    if guardar: sim.guardar_dataset(df)
    return {
        "resumen":      sim.resumen_estadistico(df),
        "muestra":      df.head(10).to_dict(orient="records"),
        "dataset_path": DATASET_PATH if guardar else None,
    }


if __name__=="__main__":
    r=ejecutar_simulacion()
    print(f"\n✅ Simulación completada — {r['resumen']['total_registros']} registros")
    for k,v in r["resumen"]["estadisticas"].items():
        print(f"   {k}: {v}")