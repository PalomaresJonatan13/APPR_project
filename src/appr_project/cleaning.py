"""Limpieza del historial de pacientes y comparación de distribuciones."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def detect_history_deviations(
    frame,
    variable,
    alpha=0.5,
    *,
    window=20,
    min_history=5,
    max_age_hours=24.0,
    k=5.0,
    min_margin=None,
):
    """Devuelve una fila por desviación, con el ajuste propuesto y su contexto.

    La referencia usa únicamente mediciones originales previas del paciente.
    Con historial insuficiente o MAD cero sin margen explícito, se omite la
    evaluación. Un min_margin positivo fija una semiamplitud mínima en las
    unidades de la variable. alpha es el peso del valor observado.
    row_position identifica la fila por posición en el dataframe de entrada.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha debe estar entre 0 y 1.")
    if any(not isinstance(n, (int, np.integer)) for n in (window, min_history)):
        raise ValueError("window y min_history deben ser enteros.")
    if not 2 <= min_history <= window:
        raise ValueError("Se requiere 2 <= min_history <= window.")
    if not np.isfinite([k, max_age_hours]).all() or min(k, max_age_hours) <= 0:
        raise ValueError("k y max_age_hours deben ser positivos y finitos.")
    if min_margin is not None and (not np.isfinite(min_margin) or min_margin <= 0):
        raise ValueError("min_margin debe ser positivo y finito.")

    columns = [
        "row_position",
        "patient_id",
        "ICULOS",
        "variable",
        "original_value",
        "history_n",
        "history_median",
        "history_mad",
        "lower",
        "upper",
        "alpha",
        "adjusted_value",
        "adjustment",
    ]
    work = frame[["patient_id", "ICULOS", variable]].copy()
    work["row_position"] = np.arange(len(frame))
    if work["patient_id"].isna().any() or not np.isfinite(work["ICULOS"]).all():
        raise ValueError("Se necesitan identificadores y horas válidas.")
    if work.duplicated(["patient_id", "ICULOS"]).any():
        raise ValueError("Hay horas repetidas dentro de un paciente.")
    work = work.loc[np.isfinite(work[variable])].sort_values(["patient_id", "ICULOS"])
    records = []
    for patient_id, group in work.groupby("patient_id", sort=False):
        if len(group) <= min_history:
            continue
        values = group[variable].to_numpy(dtype=float)
        hours = group["ICULOS"].to_numpy(dtype=float)
        previous = sliding_window_view(
            np.pad(values, (window, 0), constant_values=np.nan), window
        )[:-1]
        previous_hours = sliding_window_view(
            np.pad(hours, (window, 0), constant_values=-np.inf), window
        )[:-1]
        history = np.where(
            hours[:, None] - previous_hours <= max_age_hours, previous, np.nan
        )
        history_n = np.isfinite(history).sum(axis=1)
        positions = np.flatnonzero(history_n >= min_history)
        if not len(positions):
            continue
        history = history[positions]
        median = np.nanmedian(history, axis=1)
        mad = np.nanmedian(np.abs(history - median[:, None]), axis=1)
        margin = np.maximum(k * mad / 0.6745, min_margin or 0)
        lower, upper = median - margin, median + margin
        observed = values[positions]
        boundary = np.clip(observed, lower, upper)
        flagged = (margin > 0) & (observed != boundary)
        if not flagged.any():
            continue
        adjusted = boundary[flagged] + alpha * (observed[flagged] - boundary[flagged])
        rows = positions[flagged]
        records.append(
            pd.DataFrame(
                {
                    "row_position": group["row_position"].to_numpy()[rows],
                    "patient_id": patient_id,
                    "ICULOS": hours[rows],
                    "variable": variable,
                    "original_value": observed[flagged],
                    "history_n": history_n[rows],
                    "history_median": median[flagged],
                    "history_mad": mad[flagged],
                    "lower": lower[flagged],
                    "upper": upper[flagged],
                    "alpha": alpha,
                    "adjusted_value": adjusted,
                    "adjustment": adjusted - observed[flagged],
                }
            )
        )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.concat(records, ignore_index=True).sort_values(
        "row_position", ignore_index=True
    )


def proporcion_ab(eventos, validos, *, pacientes_ab, pesos_ab):
    """Estima la distancia entre proporciones y su intervalo por pacientes."""
    tasas, replicas = [], []
    for evento, valido, pacientes, pesos in zip(eventos, validos, pacientes_ab, pesos_ab):
        n = pesos.shape[1]
        conteos = np.bincount(pacientes, weights=evento & valido, minlength=n)
        totales = np.bincount(pacientes, weights=valido, minlength=n)
        tasas.append(conteos.sum() / totales.sum() if totales.sum() else np.nan)
        totales_boot = pesos @ totales.astype(np.float32)
        replicas.append(np.divide(
            pesos @ conteos.astype(np.float32), totales_boot,
            out=np.full(pesos.shape[0], np.nan), where=totales_boot > 0,
        ))
    diferencia = tasas[0] - tasas[1]
    radio = np.quantile(np.abs(replicas[0] - replicas[1] - diferencia), 0.95)
    distancia = abs(diferencia)
    return distancia, distancia - radio, distancia + radio, *tasas


def ks_ab(observaciones, validos, *, pacientes_ab, pesos_ab):
    """Calcula KS y sus límites aproximados con bootstrap por pacientes."""
    if any(valores.size == 0 for valores in observaciones):
        return np.nan, np.nan, np.nan
    ordenadas = [np.sort(valores) for valores in observaciones]
    soporte = np.union1d(*ordenadas)
    a, b = [
        np.searchsorted(valores, soporte, side="right") / valores.size
        for valores in ordenadas
    ]
    distancia = np.max(np.abs(a - b))
    cortes = np.unique(np.quantile(np.concatenate(ordenadas), np.linspace(0, 1, 101)))

    acumuladas, errores = [], []
    for valores, valido, pacientes, pesos in zip(
        observaciones, validos, pacientes_ab, pesos_ab
    ):
        # Cada corte tiene su propio compartimento para distinguir < de <=.
        posicion = np.searchsorted(cortes, valores)
        coincide = valores == cortes[np.minimum(posicion, cortes.size - 1)]
        compartimento = 2 * posicion + coincide
        ancho = 2 * cortes.size + 1
        conteos = np.bincount(
            pacientes[valido] * ancho + compartimento, minlength=pesos.shape[1] * ancho,
        ).reshape(-1, ancho)
        acumulada = conteos.sum(axis=0).cumsum() / valores.size
        acumuladas.append(acumulada)
        boot = (pesos @ conteos.astype(np.float32)).cumsum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            errores.append(np.max(np.abs(boot / boot[:, -1:] - acumulada), axis=1))

    a, b = acumuladas
    radio = np.quantile(errores[0] + errores[1], 0.95)
    # En cada intervalo abierto, las CDF quedan acotadas por sus extremos.
    a_inf, b_inf = np.r_[0, a[1::2]], np.r_[0, b[1::2]]
    inferior = np.max(np.abs(a - b))
    superior = max(inferior, np.max(a[::2] - b_inf), np.max(b[::2] - a_inf))
    return distancia, inferior - radio, superior + radio


def fila_ab(
    variable, medida, distancia, inferior, superior, a=np.nan, b=np.nan,
    *, margenes_ab, margenes_por_variable_ab,
):
    """Construye una fila de equivalencia con distancias en puntos porcentuales."""
    margen = margenes_por_variable_ab.get((variable, medida), margenes_ab[medida])
    inferior, superior = np.maximum(0, inferior * 100), np.minimum(100, superior * 100)
    estado = "Inconcluso"
    if superior < margen:
        estado = "Equivalente"
    elif inferior > margen:
        estado = "Diferencia relevante"
    return {
        "variable": variable, "medida": medida, "A (%)": a * 100, "B (%)": b * 100,
        "distancia (pp)": distancia * 100, "inferior 95 % (pp)": inferior,
        "superior 95 % (pp)": superior, "margen (pp)": margen, "resultado": estado,
    }
