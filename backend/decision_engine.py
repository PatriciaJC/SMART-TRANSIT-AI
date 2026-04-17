"""
decision_engine.py — Motor de decisiones basado en reglas para MiBus Panamá.
Umbrales en % de ocupación (no número fijo) para soportar capacidades variables
por familia (E=80, C=70, S/T=65, M=55).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from utils import (
    calcular_capacidad_dinamica,
    calcular_frecuencia_optima,
    calcular_indice_confiabilidad,
    get_capacidad, get_info_ruta,
    logger,
)


class EstadoBus(str,Enum):
    DISPONIBLE     = "DISPONIBLE"
    LLENO          = "BUS LLENO"
    CRITICO        = "CAPACIDAD CRÍTICA"
    FUERA_SERVICIO = "FUERA DE SERVICIO"

class EstadoServicio(str,Enum):
    NORMAL          = "SERVICIO NORMAL"
    RETRASO_LEVE    = "RETRASO LEVE"
    RETRASO         = "RETRASO"
    RETRASO_CRITICO = "RETRASO CRÍTICO"

class NivelDemanda(str,Enum):
    BAJA     = "DEMANDA BAJA"
    MODERADA = "DEMANDA MODERADA"
    ALTA     = "DEMANDA ALTA"
    MUY_ALTA = "DEMANDA MUY ALTA"

class Prioridad(str,Enum):
    BAJA    = "baja"
    MEDIA   = "media"
    ALTA    = "alta"
    CRITICA = "crítica"


class Umbrales:
    # ── CAMBIO: porcentajes en lugar de números fijos ──
    # Funciona correctamente para buses de 55, 65, 70 u 80 asientos
    OCUPACION_LLENO_PCT    = 95   # ≥95% → BUS LLENO
    OCUPACION_CRITICO_PCT  = 88   # ≥88% → CAPACIDAD CRÍTICA
    OCUPACION_ALTA_PCT     = 80
    OCUPACION_MUY_ALTA_PCT = 92

    # Confiabilidad / puntualidad (sin cambio)
    RETRASO_CRITICO = 0.50
    RETRASO         = 0.65
    RETRASO_LEVE    = 0.80

    # Tráfico (sin cambio)
    TRAFICO_ALTO    = 0.75
    TRAFICO_CRITICO = 0.90

    # Frecuencia
    FRECUENCIA_ALTA_DEMANDA = 5.0
    FRECUENCIA_BAJA_DEMANDA = 20.0


@dataclass
class EstadoSistema:
    pasajeros:            int
    capacidad_base:       int   = 65        # se sobreescribe con get_capacidad()
    densidad_trafico:     float = 0.5
    tiempo_estimado_min:  float = 15.0
    tiempo_real_min:      float = 15.0
    hora:                 int   = 12
    es_hora_pico:         bool  = False
    incidente:            bool  = False
    tipo_incidente:       str   = "ninguno"
    ruta:                 str   = "C640"
    clima:                str   = "despejado"
    capacidad_disponible: Optional[float] = None
    indice_confiabilidad: Optional[float] = None
    frecuencia_optima_min:Optional[float] = None

    def __post_init__(self):
        # Si no se pasó capacidad_base, infiere de la ruta real
        info = get_info_ruta(self.ruta)
        if info and self.capacidad_base == 65:
            self.capacidad_base = get_capacidad(self.ruta)
        if self.capacidad_disponible is None:
            self.capacidad_disponible = calcular_capacidad_dinamica(self.pasajeros,self.capacidad_base)
        if self.indice_confiabilidad is None:
            self.indice_confiabilidad = calcular_indice_confiabilidad(self.tiempo_estimado_min,self.tiempo_real_min)
        if self.frecuencia_optima_min is None:
            self.frecuencia_optima_min = calcular_frecuencia_optima(self.pasajeros,self.capacidad_base)


@dataclass
class Accion:
    codigo:      str
    descripcion: str
    prioridad:   Prioridad
    categoria:   str
    detalles:    Optional[str] = None


@dataclass
class DecisionResult:
    estado_bus:          EstadoBus
    estado_servicio:     EstadoServicio
    nivel_demanda:       NivelDemanda
    acciones:            List[Accion] = field(default_factory=list)
    alertas:             List[str]    = field(default_factory=list)
    metricas_calculadas: Dict         = field(default_factory=dict)
    resumen_ejecutivo:   str          = ""

    def as_dict(self)->Dict:
        orden=["baja","media","alta","crítica"]
        return {
            "estado_bus":      self.estado_bus.value,
            "estado_servicio": self.estado_servicio.value,
            "nivel_demanda":   self.nivel_demanda.value,
            "acciones": [
                {"codigo":a.codigo,"descripcion":a.descripcion,
                 "prioridad":a.prioridad.value,"categoria":a.categoria,"detalles":a.detalles}
                for a in sorted(self.acciones,key=lambda x:orden.index(x.prioridad.value),reverse=True)
            ],
            "alertas":             self.alertas,
            "metricas_calculadas": self.metricas_calculadas,
            "resumen_ejecutivo":   self.resumen_ejecutivo,
            "total_acciones":      len(self.acciones),
            "acciones_criticas":   sum(1 for a in self.acciones if a.prioridad==Prioridad.CRITICA),
        }


class MotorDecisiones:

    # ── CAMBIO: comparación en porcentaje, no número absoluto ──
    def _evaluar_estado_bus(self,e:EstadoSistema)->EstadoBus:
        pct=(e.pasajeros/e.capacidad_base)*100
        if pct>=Umbrales.OCUPACION_LLENO_PCT:   return EstadoBus.LLENO
        if pct>=Umbrales.OCUPACION_CRITICO_PCT: return EstadoBus.CRITICO
        return EstadoBus.DISPONIBLE

    def _evaluar_estado_servicio(self,e:EstadoSistema)->EstadoServicio:
        i=e.indice_confiabilidad
        if i<Umbrales.RETRASO_CRITICO: return EstadoServicio.RETRASO_CRITICO
        if i<Umbrales.RETRASO:         return EstadoServicio.RETRASO
        if i<Umbrales.RETRASO_LEVE:    return EstadoServicio.RETRASO_LEVE
        return EstadoServicio.NORMAL

    def _evaluar_nivel_demanda(self,e:EstadoSistema)->NivelDemanda:
        pct=(e.pasajeros/e.capacidad_base)*100
        if pct>=Umbrales.OCUPACION_MUY_ALTA_PCT: return NivelDemanda.MUY_ALTA
        if pct>=Umbrales.OCUPACION_ALTA_PCT:      return NivelDemanda.ALTA
        if pct>=40:                                return NivelDemanda.MODERADA
        return NivelDemanda.BAJA

    def _reglas_capacidad(self,e,eb,acc,alt):
        pct=round(e.pasajeros/e.capacidad_base*100,1)
        if eb==EstadoBus.LLENO:
            alt.append(f"⚠️ BUS LLENO en {e.ruta} — {pct}% ocupación ({e.pasajeros}/{e.capacidad_base})")
            acc.append(Accion("CAP-001","Enviar bus adicional",Prioridad.CRITICA,"capacidad",
                f"Ruta {e.ruta} saturada ({pct}%). Despachar unidad de refuerzo inmediatamente."))
            acc.append(Accion("CAP-002","Notificar próxima parada como alternativa",Prioridad.ALTA,"comunicación",
                "Informar a usuarios en app y pantallas sobre siguiente servicio."))
        elif eb==EstadoBus.CRITICO:
            alt.append(f"⚡ CAPACIDAD CRÍTICA en {e.ruta} — {pct}% ({e.capacidad_disponible:.0f} asientos restantes)")
            acc.append(Accion("CAP-003","Preparar bus de refuerzo en espera",Prioridad.ALTA,"capacidad",
                f"Activar reserva si ocupación supera {Umbrales.OCUPACION_LLENO_PCT}% en próxima parada."))

    def _reglas_puntualidad(self,e,es,acc,alt):
        i=e.indice_confiabilidad
        desv=e.tiempo_real_min-e.tiempo_estimado_min
        if es==EstadoServicio.RETRASO_CRITICO:
            alt.append(f"🚨 RETRASO CRÍTICO — I_cs={i:.2f} | Desvío: +{desv:.1f} min en {e.ruta}")
            acc.append(Accion("PUN-001","Reasignar ruta para reducir desvío",Prioridad.CRITICA,"puntualidad",
                f"Desvío {desv:.1f} min. Evaluar ruta alternativa. I_cs={i:.2f}"))
            acc.append(Accion("PUN-002","Solicitar prioridad semafórica",Prioridad.ALTA,"infraestructura",
                "Activar preferencia semafórica para recuperar tiempo."))
        elif es==EstadoServicio.RETRASO:
            alt.append(f"⚠️ RETRASO — I_cs={i:.2f} | Desvío: +{desv:.1f} min")
            acc.append(Accion("PUN-003","Optimizar paradas para recuperar tiempo",Prioridad.ALTA,"puntualidad",
                f"Reducir tiempo en paradas de baja demanda. Desvío: {desv:.1f} min"))
        elif es==EstadoServicio.RETRASO_LEVE:
            acc.append(Accion("PUN-004","Monitorear puntualidad",Prioridad.MEDIA,"monitoreo",
                f"I_cs={i:.2f}. Retraso leve. Mantener vigilancia."))

    def _reglas_demanda(self,e,nd,acc,alt):
        pct=round(e.pasajeros/e.capacidad_base*100,1)
        if nd==NivelDemanda.MUY_ALTA:
            alt.append(f"📈 DEMANDA MUY ALTA — {pct}% en {e.ruta} (cap. {e.capacidad_base})")
            acc.append(Accion("DEM-001","Aumentar frecuencia inmediatamente",Prioridad.CRITICA,"frecuencia",
                f"Frecuencia óptima: cada {e.frecuencia_optima_min:.1f} min. Despachar buses adicionales."))
        elif nd==NivelDemanda.ALTA:
            acc.append(Accion("DEM-002","Ajustar frecuencia por alta demanda",Prioridad.ALTA,"frecuencia",
                f"Frecuencia recomendada: cada {e.frecuencia_optima_min:.1f} min. Ocupación: {pct}%"))
        elif nd==NivelDemanda.BAJA:
            acc.append(Accion("DEM-003","Considerar reducción de frecuencia",Prioridad.BAJA,"eficiencia",
                f"Baja demanda en {e.ruta}. Intervalo sugerido: {e.frecuencia_optima_min:.1f} min."))

    def _reglas_trafico(self,e,acc,alt):
        d=e.densidad_trafico
        if d>Umbrales.TRAFICO_CRITICO:
            alt.append(f"🚗 TRÁFICO CRÍTICO — densidad {d:.0%} en {e.ruta}")
            acc.append(Accion("TRA-001","Activar ruta alternativa por congestión",Prioridad.ALTA,"tráfico",
                f"Densidad {d:.0%}. Evaluar desvío por corredor secundario."))
        elif d>Umbrales.TRAFICO_ALTO:
            acc.append(Accion("TRA-002","Ajustar tiempos estimados por tráfico",Prioridad.MEDIA,"tráfico",
                f"Densidad {d:.0%}. Actualizar ETA en app y pantallas."))

    def _reglas_incidentes(self,e,acc,alt):
        if not e.incidente: return
        alt.append(f"🚨 INCIDENTE: {e.tipo_incidente.upper()} en {e.ruta}")
        acc.append(Accion("INC-001",f"Protocolo de emergencia: {e.tipo_incidente}",Prioridad.CRITICA,"incidente",
            f"Tipo: {e.tipo_incidente}. Notificar a central de operaciones inmediatamente."))
        acc.append(Accion("INC-002","Desviar unidades afectadas a ruta alternativa",Prioridad.CRITICA,"incidente",
            f"Reasignar buses de {e.ruta} mientras se resuelve el incidente."))
        acc.append(Accion("INC-003","Informar a usuarios en tiempo real",Prioridad.ALTA,"comunicación",
            "Publicar alerta en app, SMS y pantallas de información."))

    def _reglas_hora_pico(self,e,acc,alt):
        if not e.es_hora_pico: return
        pct=e.pasajeros/e.capacidad_base*100
        alt.append(f"⏰ HORA PICO ({e.hora}:00 h) en {e.ruta}")
        if pct>70:
            acc.append(Accion("PIC-001","Reforzar flota en hora pico",Prioridad.ALTA,"hora_pico",
                f"Hora pico {e.hora}:00h. Activar buses adicionales. Frec.: cada {e.frecuencia_optima_min:.1f} min."))

    def analizar(self,estado:EstadoSistema)->DecisionResult:
        acc,alt=[],[]
        eb = self._evaluar_estado_bus(estado)
        es = self._evaluar_estado_servicio(estado)
        nd = self._evaluar_nivel_demanda(estado)
        self._reglas_capacidad(estado,eb,acc,alt)
        self._reglas_puntualidad(estado,es,acc,alt)
        self._reglas_demanda(estado,nd,acc,alt)
        self._reglas_trafico(estado,acc,alt)
        self._reglas_incidentes(estado,acc,alt)
        self._reglas_hora_pico(estado,acc,alt)
        if not acc:
            acc.append(Accion("SIS-000","Sistema operando con normalidad",Prioridad.BAJA,"monitoreo",
                "No se detectaron anomalías."))
        n_crit=sum(1 for a in acc if a.prioridad==Prioridad.CRITICA)
        resumen=(f"Ruta {estado.ruta} | {estado.hora}:00 — {eb.value} | "
                 f"{es.value} | {nd.value}"
                 +(f" | ⚠️ {n_crit} CRÍTICA(S)" if n_crit else "")
                 +(f" | 🚨 {estado.tipo_incidente}" if estado.incidente else ""))
        metricas={
            "capacidad_disponible":   estado.capacidad_disponible,
            "indice_confiabilidad":   estado.indice_confiabilidad,
            "frecuencia_optima_min":  estado.frecuencia_optima_min,
            "ocupacion_pct":          round(estado.pasajeros/estado.capacidad_base*100,2),
            "desvio_tiempo_min":      round(estado.tiempo_real_min-estado.tiempo_estimado_min,2),
        }
        return DecisionResult(eb,es,nd,acc,alt,metricas,resumen)


def analizar_estado(datos:Dict)->Dict:
    try:
        ruta = str(datos.get("ruta","C640"))
        estado=EstadoSistema(
            pasajeros=int(datos.get("pasajeros",30)),
            capacidad_base=int(datos.get("capacidad_base",get_capacidad(ruta))),
            densidad_trafico=float(datos.get("densidad_trafico",0.5)),
            tiempo_estimado_min=float(datos.get("tiempo_estimado_min",15.0)),
            tiempo_real_min=float(datos.get("tiempo_real_min",15.0)),
            hora=int(datos.get("hora",12)),
            es_hora_pico=bool(datos.get("es_hora_pico",False)),
            incidente=bool(datos.get("incidente",False)),
            tipo_incidente=str(datos.get("tipo_incidente","ninguno")),
            ruta=ruta,
            clima=str(datos.get("clima","despejado")),
            capacidad_disponible=datos.get("capacidad_disponible"),
            indice_confiabilidad=datos.get("indice_confiabilidad"),
            frecuencia_optima_min=datos.get("frecuencia_optima_min"),
        )
    except (TypeError,ValueError) as e:
        return {"error":f"Datos inválidos: {e}"}
    return MotorDecisiones().analizar(estado).as_dict()


if __name__=="__main__":
    # Prueba con ruta real S447 (capacidad 65, familia S)
    r=analizar_estado({
        "pasajeros":62,"ruta":"S447","hora":8,"es_hora_pico":True,
        "densidad_trafico":0.88,"tiempo_estimado_min":28.0,"tiempo_real_min":41.0,
        "incidente":True,"tipo_incidente":"accidente","clima":"lluvia_leve",
    })
    print(f"\nRuta S447 — Cap. real: 65 asientos")
    print(f"Estado Bus     : {r['estado_bus']}")
    print(f"Estado Servicio: {r['estado_servicio']}")
    print(f"Ocupación      : {r['metricas_calculadas']['ocupacion_pct']}%")
    print(f"Acciones críticas: {r['acciones_criticas']}")
    for a in r["acciones"][:3]:
        print(f"  [{a['prioridad'].upper()}] {a['codigo']}: {a['descripcion']}")