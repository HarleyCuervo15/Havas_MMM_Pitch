"""
pitch.py — Modelo econométrico rápido (modo pitch) como módulo del app de EDA.

Ridge con no-negatividad sobre adstock + saturación, búsqueda aleatoria de
hiperparámetros con refinamiento local, dummies sugeridas por residuales y
anclaje opcional a un benchmark de contribución.

Uso desde app.py:

    import pitch
    with t_pitch:
        pitch.render(st, df_model, date_col, kpi_col, media_cols, ctrl_cols, P)

Todo el estado vive en claves de session_state con prefijo `pitch_` para no
chocar con el resto de la app.
"""
import io
import json
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import lsq_linear

CLIENTE = "Tigo – Millicom"
NAVY, BLUE, RED, AMBER = "#00263A", "#0088CE", "#E4002B", "#E8A400"


def usar_paleta(P):
    """Toma los colores del selector de tema del app principal."""
    global NAVY, BLUE, RED
    if P:
        NAVY = P.get("primario", NAVY)
        BLUE = P.get("secundario", BLUE)
        RED = P.get("acento", RED)


# ============================================================================
# NÚCLEO ECONOMÉTRICO
# ============================================================================
def adstock_geom(x, theta, L=12):
    """Adstock geométrico normalizado (los pesos suman 1: no infla la escala)."""
    x = np.asarray(x, dtype=float)
    if theta <= 0:
        return x.copy()
    w = theta ** np.arange(L + 1)
    w /= w.sum()
    return np.convolve(x, w)[: len(x)]


def saturar(a, alpha, tipo="potencia", gamma=0.5, ref=None):
    """
    Curva de respuesta cóncava sobre la serie ya adstockeada.
    `ref` fija la escala (máximo histórico). Es indispensable para que las
    curvas de respuesta a 1.5x / 2x de inversión no salgan planas.
    """
    a = np.asarray(a, dtype=float)
    if ref is None:
        ref = a.max()
    if ref <= 0:
        return np.zeros_like(a), 1.0
    z = a / ref
    if tipo == "hill":
        g = max(gamma, 1e-3)
        s = (z ** alpha) / (z ** alpha + g ** alpha)
        s = s * (1 + g ** alpha)          # normaliza a 1 cuando z = 1
    else:
        s = np.power(np.clip(z, 0, None), alpha)
    return s, ref


def construir_controles(n, usar_tendencia, usar_cuadratica, n_fourier, periodo):
    """Tendencia + estacionalidad de Fourier. Reemplaza los knots de Meridian."""
    cols, nombres = [], []
    t = np.arange(n) / max(n - 1, 1)
    if usar_tendencia:
        cols.append(t); nombres.append("tendencia")
    if usar_cuadratica:
        cols.append(t ** 2); nombres.append("tendencia_2")
    idx = np.arange(n)
    for k in range(1, n_fourier + 1):
        cols.append(np.sin(2 * np.pi * k * idx / periodo)); nombres.append(f"sin_{k}")
        cols.append(np.cos(2 * np.pi * k * idx / periodo)); nombres.append(f"cos_{k}")
    if not cols:
        return np.zeros((n, 0)), []
    return np.column_stack(cols), nombres


def ajustar_ridge_acotado(X, y, lam, pos_mask):
    """
    Ridge con restricción de no-negatividad selectiva.
    Penalización por matriz de diseño aumentada + cotas resueltas con mínimos
    cuadrados acotados: problema convexo, converge siempre y en milisegundos.
    """
    n, p = X.shape
    mx, sx = X.mean(0), X.std(0)
    sx = np.where(sx <= 1e-12, 1.0, sx)
    my, sy = y.mean(), y.std()
    sy = sy if sy > 1e-12 else 1.0
    Xs = (X - mx) / sx
    ys = (y - my) / sy
    A = np.vstack([Xs, np.sqrt(lam * n) * np.eye(p)])
    b = np.concatenate([ys, np.zeros(p)])
    lo = np.where(pos_mask, 0.0, -np.inf)
    hi = np.full(p, np.inf)
    try:
        r = lsq_linear(A, b, bounds=(lo, hi), method="trf", lsq_solver="exact",
                       tol=1e-10, max_iter=120)
        beta_s = r.x
    except Exception:
        beta_s = np.zeros(p)
    beta = beta_s * sy / sx
    return my - mx @ beta, beta


def metricas(y, yhat, p):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    n = len(y)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / sst if sst > 0 else np.nan
    gl = n - p - 1
    r2a = 1 - (1 - r2) * (n - 1) / gl if gl > 0 else np.nan
    rmse = np.sqrt(sse / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs((y - yhat) / np.where(y == 0, np.nan, y)))) * 100
    e = y - yhat
    dw = float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2)) if np.sum(e ** 2) > 0 else np.nan
    return {"R2": r2, "R2_adj": r2a, "NRMSE": rmse / y.mean() if y.mean() else np.nan,
            "MAPE": mape, "DW": dw}


def _r2_oos(y_test, pred, ref):
    """R² fuera de muestra contra el predictor ingenuo (media del entrenamiento)."""
    sst = float(np.sum((y_test - ref) ** 2))
    return 1 - float(np.sum((y_test - pred) ** 2)) / sst if sst > 0 else np.nan


# ============================================================================
# DUMMIES
# ============================================================================
def construir_dummies(fechas, specs):
    """Convierte especificaciones de dummies en columnas 0/1."""
    f = pd.DatetimeIndex(pd.Series(fechas).values)
    n = len(f)
    cols, nombres = [], []
    for s in specs:
        v = np.zeros(n)
        t = s["tipo"]
        if t == "punto":
            for d in s["fechas"]:
                v[f == pd.Timestamp(d)] = 1.0
        elif t == "rango":
            v[(f >= pd.Timestamp(s["inicio"])) & (f <= pd.Timestamp(s["fin"]))] = 1.0
        elif t == "escalon":
            v[f >= pd.Timestamp(s["desde"])] = 1.0
        elif t == "mes":
            v[f.month == int(s["mes"])] = 1.0
        elif t == "semana_mes":
            v[f.day <= 7] = 1.0
        if v.sum() == 0 or v.sum() == n:
            continue
        cols.append(v)
        nombres.append(s["nombre"])
    if not cols:
        return np.zeros((n, 0)), []
    return np.column_stack(cols), nombres


def _oos(X, y, lam, pos_mask, h, pliegues):
    """R² fuera de muestra promedio para un diseño dado."""
    if h <= 0:
        return np.nan
    n = len(y)
    puntajes = []
    for f in range(pliegues, 0, -1):
        fin = n - h * (f - 1)
        ini = fin - h
        if ini - h < 20:
            continue
        b0, beta = ajustar_ridge_acotado(X[:ini], y[:ini], lam, pos_mask)
        puntajes.append(_r2_oos(y[ini:fin], b0 + X[ini:fin] @ beta, float(y[:ini].mean())))
    return float(np.mean(puntajes)) if puntajes else np.nan


def _pct_medios(X, y, b, k):
    return float(sum(b[j] * X[:, j].sum() for j in range(k)) / y.sum() * 100)


def ganancia_dummy(X, y, lam, pos_mask, nueva, h=0, pliegues=1, k=0):
    """
    ΔR², ΔR² ajustado y ΔR² fuera de muestra al añadir una columna (o bloque).
    El delta fuera de muestra es el que distingue una dummy que EXPLICA algo de
    una que solo tapa una observación incómoda.
    """
    nueva = np.atleast_2d(nueva)
    if nueva.shape[0] != X.shape[0]:
        nueva = nueva.T
    b0, b = ajustar_ridge_acotado(X, y, lam, pos_mask)
    m0 = metricas(y, b0 + X @ b, X.shape[1])
    o0 = _oos(X, y, lam, pos_mask, h, pliegues)
    X2 = np.hstack([X, nueva])
    pos2 = np.concatenate([pos_mask, np.zeros(nueva.shape[1], bool)])
    b0b, bb = ajustar_ridge_acotado(X2, y, lam, pos2)
    m1 = metricas(y, b0b + X2 @ bb, X2.shape[1])
    o1 = _oos(X2, y, lam, pos2, h, pliegues)
    d_out = (o1 - o0) if (np.isfinite(o0) and np.isfinite(o1)) else np.nan
    d_med = (_pct_medios(X2, y, bb, k) - _pct_medios(X, y, b, k)) if k else np.nan
    return m1["R2"] - m0["R2"], m1["R2_adj"] - m0["R2_adj"], m1["R2"], d_out, d_med


def sugerir_dummies(fechas, resid, z_pico=2.5, z_racha=1.2, min_racha=3):
    """
    Propone dummies a partir de los residuales del modelo actual:
    picos atípicos, tramos con sesgo sostenido, un escalón de nivel y meses
    sistemáticamente mal explicados.
    """
    f = pd.DatetimeIndex(pd.Series(fechas).values)
    n = len(resid)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    esc = 1.4826 * mad if mad > 1e-9 else (resid.std() or 1.0)
    z = (resid - med) / esc
    specs = []

    # 1) picos individuales
    for i in np.argsort(-np.abs(z))[:8]:
        if abs(z[i]) < z_pico:
            continue
        specs.append({"tipo": "punto", "fechas": [str(f[i].date())],
                      "nombre": f"D_pico_{f[i]:%Y%m%d}",
                      "motivo": f"residual atípico (z={z[i]:.1f}) el {f[i]:%Y-%m-%d}"})

    # 2) tramos con sesgo sostenido del mismo signo
    signo = np.sign(z) * (np.abs(z) > z_racha)
    i = 0
    while i < n:
        if signo[i] != 0:
            j = i
            while j + 1 < n and signo[j + 1] == signo[i]:
                j += 1
            if j - i + 1 >= min_racha:
                lado = "sobre" if signo[i] > 0 else "sub"
                specs.append({"tipo": "rango", "inicio": str(f[i].date()), "fin": str(f[j].date()),
                              "nombre": f"D_periodo_{f[i]:%Y%m%d}",
                              "motivo": f"{j-i+1} periodos seguidos {lado}estimados "
                                        f"({f[i]:%Y-%m-%d} a {f[j]:%Y-%m-%d})"})
            i = j + 1
        else:
            i += 1

    # 3) escalón de nivel: corte que más separa las medias de los residuales
    mejor, corte = 0.0, None
    for k in range(max(8, n // 10), n - max(8, n // 10)):
        d = abs(resid[k:].mean() - resid[:k].mean())
        if d > mejor:
            mejor, corte = d, k
    if corte is not None and mejor > 0.5 * esc:
        specs.append({"tipo": "escalon", "desde": str(f[corte].date()),
                      "nombre": f"D_escalon_{f[corte]:%Y%m%d}",
                      "motivo": f"cambio de nivel del residual desde {f[corte]:%Y-%m-%d}"})

    # 4) meses sistemáticamente mal explicados
    dfm = pd.DataFrame({"mes": f.month, "z": z})
    for mes, g in dfm.groupby("mes"):
        if len(g) >= 3 and abs(g["z"].mean()) > 0.7:
            specs.append({"tipo": "mes", "mes": int(mes), "nombre": f"D_mes_{int(mes):02d}",
                          "motivo": f"el mes {int(mes)} queda sesgado en promedio (z={g['z'].mean():.2f})"})
    return specs


def evaluar_sugerencias(fechas, X, y, lam, pos_mask, specs, h=0, pliegues=1, k=0):
    """Ordena las dummies candidatas por cuánto suben el R² dentro y fuera de muestra."""
    filas = []
    for s in specs:
        col, _ = construir_dummies(fechas, [s])
        if col.shape[1] == 0:
            continue
        d_r2, d_adj, r2_nuevo, d_out, d_med = ganancia_dummy(X, y, lam, pos_mask, col,
                                                             h, pliegues, k)
        if d_adj <= 0 or (np.isfinite(d_out) and d_out < -0.002):
            veredicto = "solo tapa el dato"
        elif np.isfinite(d_med) and abs(d_med) > 4:
            veredicto = "mueve la descomposición"
        else:
            veredicto = "explica"
        filas.append({"dummy": s["nombre"], "tipo": s["tipo"], "motivo": s["motivo"],
                      "ΔR2": round(d_r2, 4), "ΔR2_adj": round(d_adj, 4),
                      "ΔR2_fuera": None if not np.isfinite(d_out) else round(d_out, 4),
                      "Δmedios_pp": None if not np.isfinite(d_med) else round(d_med, 2),
                      "veredicto": veredicto,
                      "R2_resultante": round(r2_nuevo, 4), "periodos_marcados": int(col.sum()),
                      "_spec": s})
    if not filas:
        return pd.DataFrame(columns=["dummy", "tipo", "motivo", "ΔR2", "ΔR2_adj", "ΔR2_fuera",
                                     "Δmedios_pp", "veredicto", "R2_resultante",
                                     "periodos_marcados"]), []
    orden = {"explica": 0, "mueve la descomposición": 1, "solo tapa el dato": 2}
    df = pd.DataFrame(filas)
    df["_o"] = df["veredicto"].map(orden)
    df = df.sort_values(["_o", "ΔR2_adj"], ascending=[True, False]).drop(columns="_o")
    df = df.reset_index(drop=True)
    return df.drop(columns=["_spec"]), df["_spec"].tolist()


# ============================================================================
# RECOMENDADORES
# ============================================================================
def recomendar_especificacion(y, n, periodo, max_fourier=4):
    """Elige tendencia y número de armónicos por R² ajustado."""
    def r2adj(X):
        A = np.column_stack([np.ones(n), X]) if X.size else np.ones((n, 1))
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return metricas(y, A @ coef, A.shape[1] - 1)["R2_adj"]

    t = (np.arange(n) / max(n - 1, 1)).reshape(-1, 1)
    base_lin = t
    base_cua = np.hstack([t, t ** 2])
    usar_cuad = r2adj(base_cua) > r2adj(base_lin) + 0.005
    base = base_cua if usar_cuad else base_lin

    mejor_k, mejor_v = 0, r2adj(base)
    idx = np.arange(n)
    for k in range(1, max_fourier + 1):
        F = np.column_stack([f(2 * np.pi * m * idx / periodo)
                             for m in range(1, k + 1) for f in (np.sin, np.cos)])
        v = r2adj(np.hstack([base, F]))
        if v > mejor_v + 0.004:
            mejor_k, mejor_v = k, v
    return {"tendencia_cuadratica": bool(usar_cuad), "armonicos": int(mejor_k),
            "r2_estructura": round(float(mejor_v), 3)}


def _mejor_rezago(x, r, max_lag=8):
    """Rezago con mayor correlación entre la inversión y el residual del KPI."""
    sx = pd.Series(x); sr = pd.Series(r)
    mejor_l, mejor_c = 0, 0.0
    for L in range(max_lag + 1):
        c = sx.shift(L).corr(sr)
        if pd.notna(c) and abs(c) > abs(mejor_c):
            mejor_c, mejor_l = c, L
    return mejor_l


def recomendar_hiperparametros(mat, y, C, medios, L=12, tipo="potencia", pasadas=3):
    """
    Barrido por canal con backfitting: en cada pasada se busca (theta, alpha)
    contra el residual del KPI una vez descontada la base estructural Y lo que
    ya explican los demás canales. Sin ese descuento, dos canales colineales se
    recomiendan mutuamente el mismo perfil y los rangos salen sesgados.
    """
    n, k = mat.shape
    A = np.column_stack([np.ones(n), C]) if C.size else np.ones((n, 1))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r_total = y - A @ coef
    # winsoriza: una promo atípica sin dummy arrastra todo el barrido
    lo_w, hi_w = np.percentile(r_total, [2, 98])
    r_total = np.clip(r_total, lo_w, hi_w)

    thetas = np.round(np.arange(0.0, 0.90, 0.05), 2)
    alphas = np.round(np.arange(0.20, 1.01, 0.05), 2)
    ajuste = np.zeros((n, k))
    th_op = np.zeros(k)
    al_op = np.full(k, 0.6)
    corr_op = np.zeros(k)
    orden = np.argsort(-mat.sum(0))          # de mayor a menor inversión

    for _ in range(pasadas):
        for j in orden:
            parcial = r_total - (ajuste.sum(1) - ajuste[:, j])
            parcial = parcial - parcial.mean()
            mejor = (-9.0, th_op[j], al_op[j], np.zeros(n))
            for t in thetas:
                a = adstock_geom(mat[:, j], t, L)
                for al in alphas:
                    s, _ = saturar(a, al, tipo)
                    sd = s.std()
                    if sd < 1e-12:
                        continue
                    sc = s - s.mean()
                    b = float(sc @ parcial) / float(sc @ sc)
                    if b <= 0:                # solo efectos positivos
                        continue
                    corr = float(np.corrcoef(sc, parcial)[0, 1])
                    if np.isfinite(corr) and corr > mejor[0]:
                        mejor = (corr, float(t), float(al), b * sc)
            corr_op[j], th_op[j], al_op[j], ajuste[:, j] = mejor

    filas, rangos, inicios, reactivos = [], {}, {}, []
    soi = mat.sum(0) / max(mat.sum(), 1e-9) * 100
    for j, c in enumerate(medios):
        x = mat[:, j]
        cru = float(pd.Series(x).corr(pd.Series(y)))
        # inversión de hoy contra el KPI de ayer: si es muy negativa, el
        # presupuesto está reaccionando al desempeño en vez de causarlo
        reac = float(pd.Series(x[1:]).reset_index(drop=True)
                     .corr(pd.Series(y[:-1]).reset_index(drop=True)))
        corr, th, al = corr_op[j], th_op[j], al_op[j]
        ceros = float((x == 0).mean() * 100)
        cv = float(x.std() / x.mean()) if x.mean() > 0 else np.nan
        debil = corr < 0.12
        ancho_t = 0.35 if debil else 0.30
        ancho_a = 0.35 if debil else 0.30
        rt = (round(max(th - ancho_t, 0.0), 2), round(min(th + ancho_t, 0.95), 2))
        ra = (round(max(al - ancho_a, 0.10), 2), round(min(al + ancho_a, 1.0), 2))
        # si hay evidencia de rezago, no cerrar la puerta a un adstock largo
        lag = _mejor_rezago(x, r_total)
        if lag >= 2 and rt[1] < 0.70:
            rt = (rt[0], 0.70)
        if np.isfinite(reac) and reac < -0.30:
            nota = "⚠ posible inversión reactiva: el presupuesto sube cuando el KPI cae"
            reactivos.append(c)
        elif corr < 0.05:
            nota = "señal débil: candidato a agruparse con otro canal"
        elif ceros > 40:
            nota = "intermitente: el adstock será inestable"
        elif soi[j] > 35 and al > 0.85:
            nota = "mucho peso y poca saturación aparente: amplía el rango de alpha hacia abajo"
        else:
            nota = "ok"
        filas.append({"canal": c, "theta_inicial": round(float(th), 2),
                      "alpha_inicial": round(float(al), 2),
                      "corr_ajustada": round(float(corr), 3),
                      "corr_cruda": round(cru, 3) if np.isfinite(cru) else None,
                      "corr_reactiva": round(reac, 3) if np.isfinite(reac) else None,
                      "SOI_%": round(float(soi[j]), 1),
                      "%ceros": round(ceros, 1),
                      "CV_inversión": round(cv, 2) if np.isfinite(cv) else None,
                      "rango_theta": f"{rt[0]}–{rt[1]}", "rango_alpha": f"{ra[0]}–{ra[1]}",
                      "nota": nota})
        rangos[c] = {"theta": rt, "alpha": ra}
        inicios[c] = {"theta": round(float(th), 2), "alpha": round(float(al), 2)}
    return pd.DataFrame(filas), rangos, inicios, reactivos


# ============================================================================
# BÚSQUEDA
# ============================================================================
class Buscador:
    def __init__(self, mat_medios, C, y, tipo_sat, L, holdout, pliegues=1):
        self.mat = mat_medios
        self.C = C
        self.y = y
        self.n, self.k = mat_medios.shape
        self.tipo = tipo_sat
        self.L = L
        self.h = holdout
        self.pliegues = max(int(pliegues), 1)
        self.pos_mask = np.concatenate([np.ones(self.k, bool), np.zeros(C.shape[1], bool)])
        self.cache = {}

    def _adstock(self, j, theta):
        key = (j, round(float(theta), 3))
        if key not in self.cache:
            self.cache[key] = adstock_geom(self.mat[:, j], key[1], self.L)
        return self.cache[key]

    def diseno(self, thetas, alphas, gammas, refs=None):
        M = np.zeros((self.n, self.k))
        R = np.zeros(self.k)
        for j in range(self.k):
            a = self._adstock(j, thetas[j])
            r = None if refs is None else refs[j]
            s, r_used = saturar(a, alphas[j], self.tipo, gammas[j], r)
            M[:, j] = s
            R[j] = r_used
        return np.hstack([M, self.C]), R

    def evaluar(self, thetas, alphas, gammas, lam):
        X, refs = self.diseno(thetas, alphas, gammas)
        r2_out = np.nan
        if self.h > 0:
            puntajes = []
            for f in range(self.pliegues, 0, -1):
                fin = self.n - self.h * (f - 1)
                ini = fin - self.h
                if ini - self.h < 20:
                    continue
                b0, beta = ajustar_ridge_acotado(X[:ini], self.y[:ini], lam, self.pos_mask)
                puntajes.append(_r2_oos(self.y[ini:fin], b0 + X[ini:fin] @ beta,
                                        float(self.y[:ini].mean())))
            if puntajes:
                r2_out = float(np.mean(puntajes))

        b0f, betaf = ajustar_ridge_acotado(X, self.y, lam, self.pos_mask)
        yhat = b0f + X @ betaf
        m = metricas(self.y, yhat, X.shape[1])
        contrib = np.array([betaf[j] * X[:, j].sum() for j in range(self.k)])
        pct = contrib / self.y.sum() * 100
        return {"thetas": np.asarray(thetas, float).copy(), "alphas": np.asarray(alphas, float).copy(),
                "gammas": np.asarray(gammas, float).copy(), "lam": lam,
                "R2": m["R2"], "R2_adj": m["R2_adj"], "R2_out": r2_out, "NRMSE": m["NRMSE"],
                "MAPE": m["MAPE"], "DW": m["DW"], "pct_medios": float(pct.sum()),
                "pct_canal": pct, "b0": b0f, "beta": betaf, "refs": refs,
                "n_cero": int(np.sum(betaf[: self.k] <= 1e-12))}

    def objetivo(self, c, modo, banda, bench=None):
        if modo == "R² ajustado (dentro de muestra)":
            v = c["R2_adj"]
        elif modo == "R² fuera de muestra":
            v = c["R2_out"]
        else:
            fuera = c["R2_out"] if np.isfinite(c["R2_out"]) else c["R2_adj"]
            v = 0.5 * c["R2_adj"] + 0.5 * fuera
        if not np.isfinite(v):
            return -1e9
        if banda is not None:
            lo, hi = banda
            v -= 0.01 * (max(lo - c["pct_medios"], 0) + max(c["pct_medios"] - hi, 0))
        if bench and bench.get("peso", 0) > 0:
            desv = []
            if bench.get("total") is not None:
                desv.append(abs(c["pct_medios"] - bench["total"]))
            porc = bench.get("canal") or {}
            if porc:
                for j, obj in porc.items():
                    desv.append(abs(c["pct_canal"][j] - obj))
            if desv:
                v -= bench["peso"] * float(np.mean(desv)) / 10.0
        return v


def buscar(bus, rangos, n_iter, modo, banda, semilla, refinar,
           bench=None, inicios=None, anclar=None, progreso=None):
    """
    inicios : lista de dicts {"thetas": [...], "alphas": [...]} como puntos de partida.
    anclar  : radio alrededor del primer punto inicial para muestrear (None = rango completo).
    """
    rng = np.random.default_rng(semilla)
    k = bus.k
    th_lo = np.array([rangos[c]["theta"][0] for c in rangos], float)
    th_hi = np.array([rangos[c]["theta"][1] for c in rangos], float)
    al_lo = np.array([rangos[c]["alpha"][0] for c in rangos], float)
    al_hi = np.array([rangos[c]["alpha"][1] for c in rangos], float)

    if anclar and inicios:
        t0 = np.asarray(inicios[0]["thetas"], float)
        a0 = np.asarray(inicios[0]["alphas"], float)
        th_lo = np.clip(np.maximum(th_lo, t0 - anclar), 0.0, 0.95)
        th_hi = np.clip(np.minimum(th_hi, t0 + anclar), 0.0, 0.95)
        al_lo = np.clip(np.maximum(al_lo, a0 - anclar), 0.05, 1.0)
        al_hi = np.clip(np.minimum(al_hi, a0 + anclar), 0.05, 1.0)
        th_hi = np.maximum(th_hi, th_lo)
        al_hi = np.maximum(al_hi, al_lo)

    cand = []
    gam = np.full(k, 0.5)
    for p in (inicios or []):
        for lam in (1e-3, 1e-2, 1e-1):
            c = bus.evaluar(np.asarray(p["thetas"], float), np.asarray(p["alphas"], float), gam, lam)
            c["score"] = bus.objetivo(c, modo, banda, bench)
            c["origen"] = "punto inicial"
            cand.append(c)

    for i in range(n_iter):
        thetas = np.round(rng.uniform(th_lo, th_hi), 2)
        alphas = np.round(rng.uniform(al_lo, al_hi), 2)
        gammas = np.round(rng.uniform(0.2, 0.8, k), 2) if bus.tipo == "hill" else gam
        lam = 10 ** rng.uniform(-4.0, 0.5)
        c = bus.evaluar(thetas, alphas, gammas, lam)
        c["score"] = bus.objetivo(c, modo, banda, bench)
        c["origen"] = "búsqueda"
        cand.append(c)
        if progreso is not None and i % max(n_iter // 40, 1) == 0:
            progreso(i / n_iter * (0.75 if refinar else 1.0))

    cand.sort(key=lambda d: -d["score"])
    mejor = cand[0]

    if refinar:
        for it, paso in enumerate((0.10, 0.05)):
            for j in range(k):
                for attr, lo, hi in (("thetas", th_lo[j], th_hi[j]), ("alphas", al_lo[j], al_hi[j])):
                    for d in (-paso, paso):
                        v = mejor[attr].copy()
                        v[j] = float(np.clip(round(v[j] + d, 2), lo, hi))
                        if v[j] == mejor[attr][j]:
                            continue
                        args = {"thetas": mejor["thetas"], "alphas": mejor["alphas"],
                                "gammas": mejor["gammas"], "lam": mejor["lam"]}
                        args[attr] = v
                        c = bus.evaluar(args["thetas"], args["alphas"], args["gammas"], args["lam"])
                        c["score"] = bus.objetivo(c, modo, banda, bench)
                        c["origen"] = "refinamiento"
                        if c["score"] > mejor["score"]:
                            mejor = c
            for f in (0.3, 0.6, 1.7, 3.0):
                c = bus.evaluar(mejor["thetas"], mejor["alphas"], mejor["gammas"], mejor["lam"] * f)
                c["score"] = bus.objetivo(c, modo, banda, bench)
                c["origen"] = "refinamiento"
                if c["score"] > mejor["score"]:
                    mejor = c
            if progreso is not None:
                progreso(0.75 + 0.25 * (it + 1) / 2)
        cand.insert(0, mejor)
    if progreso is not None:
        progreso(1.0)
    return mejor, cand[:15]


# ============================================================================
# RESULTADOS
# ============================================================================
def descomponer(bus, mejor, medios, nombres_ctrl, y):
    X, _ = bus.diseno(mejor["thetas"], mejor["alphas"], mejor["gammas"], mejor["refs"])
    beta, b0, k = mejor["beta"], mejor["b0"], bus.k
    aportes = {c: beta[j] * X[:, j] for j, c in enumerate(medios)}
    base = np.full(len(y), b0)
    for j, _ in enumerate(nombres_ctrl):
        base = base + beta[k + j] * X[:, k + j]
    yhat = base + (np.sum(list(aportes.values()), axis=0) if aportes else 0)
    return aportes, base, yhat, X


def tabla_canales(medios, aportes, mat_inv, y, mejor, valor_unitario=None, benchmark=None):
    filas = []
    for j, c in enumerate(medios):
        inv = float(mat_inv[:, j].sum())
        ap = float(aportes[c].sum())
        fila = {"canal": c,
                "theta (adstock)": round(float(mejor["thetas"][j]), 2),
                "alpha (saturación)": round(float(mejor["alphas"][j]), 2),
                "coeficiente": round(float(mejor["beta"][j]), 1),
                "inversión": round(inv, 0),
                "aporte_KPI": round(ap, 0),
                "contrib_%": round(ap / y.sum() * 100, 2),
                "KPI_por_1000": round(ap / inv * 1000, 3) if inv > 0 else np.nan}
        if benchmark and j in benchmark:
            fila["benchmark_%"] = benchmark[j]
            fila["desvío_pp"] = round(fila["contrib_%"] - benchmark[j], 2)
        if valor_unitario:
            fila["ROI"] = round(ap * valor_unitario / inv, 2) if inv > 0 else np.nan
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("contrib_%", ascending=False).reset_index(drop=True)


def curva_respuesta(bus, mejor, j, factores):
    x, ref = bus.mat[:, j], mejor["refs"][j]
    out = []
    for f in factores:
        a = adstock_geom(x * f, mejor["thetas"][j], bus.L)
        s, _ = saturar(a, mejor["alphas"][j], bus.tipo, mejor["gammas"][j], ref)
        out.append(float(mejor["beta"][j] * s.sum()))
    return np.array(out)


def diagnosticar(mejor, medios, tabla, banda, umbral_r2=0.50, n_dummies=0):
    av = []
    def a(sev, tema, msg, fix):
        av.append({"sev": sev, "tema": tema, "detalle": msg, "sugerencia": fix})

    if np.isfinite(mejor["R2"]) and mejor["R2"] < umbral_r2:
        falta = (umbral_r2 - mejor["R2"]) * 100
        a("error", f"R² por debajo del mínimo ({umbral_r2*100:.0f}%)",
          f"El modelo explica {mejor['R2']*100:.1f}% de la varianza: faltan {falta:.1f} puntos. "
          f"MAPE = {mejor['MAPE']:.1f}%.",
          "Empieza por la pestaña Dummies: aplica las que más suben el R². Si el KPI es casi plano "
          "(poca varianza), mira el MAPE antes de descartar el modelo: un R² bajo con MAPE de 2–3% "
          "sigue siendo útil para dimensionar." + ("" if n_dummies else " Aún no has aplicado ninguna dummy."))
    elif np.isfinite(mejor["R2"]) and mejor["R2"] < umbral_r2 + 0.10:
        a("warning", "R² justo sobre el mínimo",
          f"R² = {mejor['R2']*100:.1f}%, apenas por encima del umbral de {umbral_r2*100:.0f}%.",
          "Revisa las dummies sugeridas y el número de armónicos antes de llevarlo a cliente.")

    lo, hi = banda
    if mejor["pct_medios"] < lo:
        a("warning", "Contribución baja",
          f"Los medios explican {mejor['pct_medios']:.1f}% del KPI, por debajo de la banda ({lo}–{hi}%).",
          "Revisa si la tendencia se está comiendo el efecto: baja los armónicos o la tendencia cuadrática.")
    elif mejor["pct_medios"] > hi:
        a("warning", "Contribución alta",
          f"Los medios explican {mejor['pct_medios']:.1f}%, por encima de la banda ({lo}–{hi}%).",
          "Suele indicar controles insuficientes: agrega precio, distribución o competencia.")
    if mejor["n_cero"] > 0:
        ceros = tabla[tabla["coeficiente"] <= 1e-9]["canal"].tolist()
        a("info", "Canales en cero",
          f"La restricción de no-negatividad dejó en cero a: {', '.join(ceros)}.",
          "Sin la restricción saldrían negativos. Suele ser colinealidad o inversión reactiva.")
    if np.isfinite(mejor["DW"]) and (mejor["DW"] < 1.5 or mejor["DW"] > 2.6):
        a("warning", "Residuales autocorrelacionados",
          f"Durbin-Watson = {mejor['DW']:.2f} (lo sano está entre 1.5 y 2.5).",
          "Falta estructura temporal: suma un armónico o una dummy de periodo.")
    if np.isfinite(mejor["R2_out"]) and mejor["R2_out"] < 0:
        a("error", "No generaliza",
          f"El R² fuera de muestra es negativo ({mejor['R2_out']:.2f}): predice peor que la media.",
          "Sobreajuste. Sube la regularización, quita dummies puntuales o reduce canales.")
    if (np.isfinite(mejor["R2"]) and np.isfinite(mejor["R2_out"])
            and mejor["R2"] - mejor["R2_out"] > 0.35 and n_dummies > 0):
        a("warning", "El ajuste no se traslada fuera de muestra",
          f"R² dentro de muestra {mejor['R2']:.2f} vs {mejor['R2_out']:.2f} fuera. "
          f"Tienes {n_dummies} dummies aplicadas.",
          "Revisa la columna ΔR2_fuera en la pestaña Dummies y quita las marcadas como "
          "'solo tapa el dato': suben el R² absorbiendo una observación, no explicando el KPI.")
    if np.isfinite(mejor["R2"]) and np.isfinite(mejor["R2_adj"]) and (mejor["R2"] - mejor["R2_adj"]) > 0.12:
        a("warning", "Demasiados parámetros",
          f"R² {mejor['R2']:.3f} vs R² ajustado {mejor['R2_adj']:.3f}: la brecha es grande.",
          "Pocas observaciones para tantas variables. Agrupa canales o quita dummies puntuales.")
    if "desvío_pp" in tabla.columns:
        peor = tabla.reindex(tabla["desvío_pp"].abs().sort_values(ascending=False).index).iloc[0]
        if abs(peor["desvío_pp"]) > 5:
            a("warning", "Lejos del benchmark",
              f"{peor['canal']} queda a {peor['desvío_pp']:+.1f} pp de su benchmark.",
              "Sube el peso del benchmark o acota el rango de ese canal. Si al forzarlo el R² cae "
              "mucho, los datos están diciendo que el benchmark no aplica a este periodo.")
    if not av:
        a("info", "Sin alertas", "El ajuste se ve consistente para uso direccional.",
          "Preséntalo como dimensionamiento, no como asignación final de presupuesto.")
    return av


# ============================================================================
# GRÁFICAS
# ============================================================================
def fig_ajuste(fechas, y, yhat):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(fechas, y, color=NAVY, lw=2, label="Real")
    ax.plot(fechas, yhat, color=RED, lw=1.8, ls="--", label="Estimado")
    ax.set_title("KPI real vs estimado", color=NAVY, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_residuales(fechas, y, yhat, marcas=None):
    e = y - yhat
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].axhline(0, color="#999", lw=1)
    axes[0].plot(fechas, e, color=BLUE, lw=1.4)
    if marcas is not None and len(marcas):
        axes[0].scatter(np.asarray(fechas)[marcas], e[marcas], color=RED, zorder=5, s=28)
    axes[0].set_title("Residuales en el tiempo (en rojo, candidatos a dummy)", fontsize=10)
    sns.histplot(e, kde=True, color=BLUE, ax=axes[1])
    axes[1].set_title("Distribución de residuales", fontsize=10)
    fig.tight_layout()
    return fig


def fig_area(fechas, base, aportes):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    series = [base] + [aportes[c] for c in aportes]
    etiquetas = ["Base (tendencia + controles)"] + list(aportes.keys())
    colores = ["#B9C7D0"] + list(sns.color_palette("crest", len(aportes)))
    ax.stackplot(fechas, series, labels=etiquetas, colors=colores)
    ax.set_title("Descomposición del KPI", color=NAVY, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


def fig_barras_contrib(tabla):
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(tabla) + 2))
    ypos = np.arange(len(tabla))[::-1]
    ax.barh(ypos, tabla["contrib_%"], color=BLUE, label="Modelo")
    if "benchmark_%" in tabla.columns:
        ax.scatter(tabla["benchmark_%"], ypos, color=RED, zorder=5, s=45, label="Benchmark")
        ax.legend(fontsize=8)
    ax.set_yticks(ypos, tabla["canal"])
    ax.set_xlabel("% del KPI")
    ax.set_title("Contribución por canal", color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_curvas(bus, mejor, medios, mat_inv):
    factores = np.linspace(0, 2, 21)
    n = len(medios)
    ncols = min(3, n) or 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    axes = axes.reshape(-1)
    for ax, j in zip(axes, range(n)):
        yv = curva_respuesta(bus, mejor, j, factores)
        inv = mat_inv[:, j].sum() * factores
        ax.plot(inv, yv, color=BLUE, lw=2)
        ax.axvline(mat_inv[:, j].sum(), color=RED, ls="--", lw=1.2)
        ax.set_title(medios[j], fontsize=10)
        ax.set_xlabel("inversión total"); ax.set_ylabel("aporte al KPI")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Curvas de respuesta (0x a 2x de la inversión actual)",
                 color=NAVY, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b


# ============================================================================
# EXPORTS
# ============================================================================
def export_json(config, mejor, medios, tabla, dummies, sigma_prior=0.7):
    priors = {}
    for _, r in tabla.iterrows():
        j = medios.index(r["canal"])
        eff = r["KPI_por_1000"]
        priors[r["canal"]] = {
            "adstock_theta": float(mejor["thetas"][j]),
            "saturacion_alpha": float(mejor["alphas"][j]),
            "contrib_pct": float(r["contrib_%"]),
            "kpi_por_1000": None if pd.isna(eff) else float(eff),
            "roi_prior_sugerido": None if ("ROI" not in tabla.columns or pd.isna(r.get("ROI")))
                                  else {"mu_log": round(float(np.log(max(r["ROI"], 1e-6))), 3),
                                        "sigma_log": sigma_prior},
        }
    met = {k: (None if not np.isfinite(mejor[k]) else round(float(mejor[k]), 4))
           for k in ["R2", "R2_adj", "R2_out", "NRMSE", "MAPE", "DW", "pct_medios"]}
    return ("# === Modo Pitch v2 (CSA · Tigo-Millicom) — generado el "
            f"{datetime.now():%Y-%m-%d %H:%M} ===\n"
            f"CONFIG_PITCH = {json.dumps(config, ensure_ascii=False, indent=4)}\n\n"
            f"DUMMIES_APLICADAS = {json.dumps(dummies, ensure_ascii=False, indent=4)}\n\n"
            f"PRIORS_SUGERIDOS = {json.dumps(priors, ensure_ascii=False, indent=4)}\n\n"
            f"METRICAS = {json.dumps(met, indent=4)}\n")


def informe_html(cliente, meta, mejor, tabla, avisos, figs_b64, dummies):
    color = {"error": RED, "warning": AMBER, "info": BLUE}
    etiqueta = {"error": "CRÍTICO", "warning": "REVISAR", "info": "INFO"}
    items = "".join(
        f'<div style="border-left:5px solid {color[a["sev"]]};background:#fff;padding:10px 14px;'
        f'margin:8px 0;border-radius:6px;"><b style="color:{color[a["sev"]]}">{etiqueta[a["sev"]]} · '
        f'{a["tema"]}</b><br>{a["detalle"]}<br><i style="color:#556;">➜ {a["sugerencia"]}</i></div>'
        for a in avisos)
    imgs = "".join(f'<h3>{t}</h3><img src="data:image/png;base64,{b}" style="width:100%;max-width:900px;">'
                   for t, b in figs_b64)
    lista_d = ("<ul>" + "".join(f"<li><b>{d['nombre']}</b> — {d.get('motivo','manual')}</li>"
                                for d in dummies) + "</ul>") if dummies else "<p>Ninguna.</p>"
    r2out = "—" if not np.isfinite(mejor["R2_out"]) else f"{mejor['R2_out']:.3f}"
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Modelo direccional MMM · {cliente}</title><style>
body{{font-family:Arial,Helvetica,sans-serif;color:#1b2a33;margin:0;background:#f4f7f9;}}
.wrap{{max-width:960px;margin:0 auto;padding:24px;}}
.header{{background:linear-gradient(120deg,{NAVY},{BLUE});color:#fff;padding:26px 28px;border-radius:12px;}}
.kick{{color:#FF5A6E;font-weight:800;letter-spacing:.14em;font-size:.75rem;text-transform:uppercase;}}
h1{{margin:4px 0;font-size:1.7rem;}} h2{{color:{NAVY};border-bottom:2px solid #e0e8ee;padding-bottom:6px;margin-top:28px;}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;}} th{{background:{NAVY};color:#fff;padding:6px 8px;text-align:left;}}
td{{padding:5px 8px;border-bottom:1px solid #e6edf1;}}
.cards{{display:flex;gap:12px;margin:10px 0;flex-wrap:wrap;}}
.card{{flex:1;min-width:150px;background:#fff;border-left:5px solid {BLUE};border-radius:8px;padding:12px 14px;}}
.card b{{font-size:1.4rem;color:{NAVY};}} .foot{{color:#8aa0ad;font-size:.8rem;text-align:center;margin-top:26px;}}
</style></head><body><div class="wrap">
<div class="header"><div class="kick">Consumer Science &amp; Analytics · MMM modo pitch</div>
<h1>Modelo econométrico direccional — {cliente}</h1>
<div>{meta['fecha']} · {meta['periodo']} · {meta['filas']} periodos · {meta['medios']} medios</div></div>
<h2>1 · Calidad del ajuste</h2>
<div class="cards">
<div class="card"><b>{mejor['R2']:.3f}</b><br>R²</div>
<div class="card"><b>{mejor['R2_adj']:.3f}</b><br>R² ajustado</div>
<div class="card"><b>{r2out}</b><br>R² fuera de muestra</div>
<div class="card"><b>{mejor['MAPE']:.1f}%</b><br>MAPE</div>
<div class="card"><b>{mejor['pct_medios']:.1f}%</b><br>Contribución de medios</div>
</div>
<h2>2 · Diagnóstico</h2>{items}
<h2>3 · Aporte por canal</h2>{tabla.to_html(index=False, border=0)}
<h2>4 · Dummies aplicadas</h2>{lista_d}
<h2>5 · Gráficas</h2>{imgs}
<p style="color:#556;font-size:.85rem;">Modelo frecuentista regularizado, sin priors. Úsalo para dimensionar
oportunidad y ordenar canales, no para asignar presupuesto: con medios colineales el reparto entre canales
puede moverse. Ese es el salto que resuelve el modelo bayesiano en la fase de delivery.</p>
<div class="foot">Consumer Science &amp; Analytics · Documento de trabajo para {cliente}</div>
</div></body></html>"""



# ============================================================================
# PESTAÑA PARA EL APP DE EDA
# ============================================================================
def _tabla(st, data):
    """st.dataframe compatible: width='stretch' en versiones nuevas, use_container_width en viejas."""
    try:
        st.dataframe(data, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True)


def render(st, df_model, date_col, kpi_col, media_cols, ctrl_cols, P=None, tipos=None):
    """
    Dibuja la pestaña completa del modelo rápido.

    df_model  : dataframe ya recortado por fechas (fecha + KPI + medios + control)
    media_cols: canales de inversión ya clasificados en el app
    ctrl_cols : variables de control (competencia, negocio, macro)
    P         : paleta del selector de colores {primario, secundario, acento}
    tipos     : dict opcional canal -> "Online"/"Offline" (solo informativo)
    """
    usar_paleta(P)

    if not media_cols:
        st.warning("Clasifica al menos un canal de inversión arriba para poder estimar el modelo.")
        return

    d = df_model.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, kpi_col]).sort_values(date_col).reset_index(drop=True)
    cols_num = [c for c in media_cols + ctrl_cols if c in d.columns]
    d[cols_num] = d[cols_num].fillna(0)
    n = len(d)
    if n < 25:
        st.error(f"Se necesitan al menos 25 periodos para estimar algo confiable (hay {n}). "
                 "Amplía el rango de fechas.")
        return

    fechas = d[date_col]
    y = d[kpi_col].values.astype(float)
    mat_inv = d[media_cols].values.astype(float)
    ctrl_cols = [c for c in ctrl_cols if c in d.columns]
    dias = pd.Series(fechas).diff().dt.days.median()
    periodo_def = 52.18 if dias and dias <= 10 else (12.0 if dias and dias <= 40 else 365.25)

    st.session_state.setdefault("pitch_dummies", [])
    st.session_state.setdefault("pitch_reco", None)
    # los widgets toman su valor de session_state: así el recomendador puede
    # actualizarlos aunque ya se hayan dibujado antes
    st.session_state.setdefault("p_cuad", True)
    st.session_state.setdefault("p_four", 2)
    st.session_state.setdefault("p_periodo", float(periodo_def))
    for c in media_cols:
        st.session_state.setdefault(f"p_th_{c}", (0.0, 0.85))
        st.session_state.setdefault(f"p_al_{c}", (0.25, 0.95))
        st.session_state.setdefault(f"p_th0_{c}", 0.40)
        st.session_state.setdefault(f"p_al0_{c}", 0.60)

    st.markdown("### ⚡ Modelo econométrico rápido")
    st.caption("Ridge con no-negatividad sobre adstock y saturación. Corre en segundos y sirve para "
               "dimensionar contribución y eficiencia en un pitch, no para asignar presupuesto. "
               "Usa los medios y controles que ya clasificaste arriba.")

    # ---------------- recomendador ----------------
    with st.expander("🧭 Recomendador de especificación e hiperparámetros", expanded=True):
        st.caption("Barre theta y alpha canal por canal contra el KPI limpio de tendencia y "
                   "estacionalidad. Es el sustituto práctico de los priors de Meridian.")
        if st.button("🧭 Analizar y recomendar", key="pitch_btn_reco"):
            C_prev, _ = construir_controles(n, True, True, 2, float(periodo_def))
            if ctrl_cols:
                C_prev = np.hstack([d[ctrl_cols].values.astype(float), C_prev])
            esp = recomendar_especificacion(y, n, float(periodo_def))
            tab_r, rangos_r, inicios_r, reactivos = recomendar_hiperparametros(
                mat_inv, y, C_prev, media_cols, 12, "potencia")
            st.session_state["pitch_reco"] = {"esp": esp, "tabla": tab_r, "rangos": rangos_r,
                                              "inicios": inicios_r, "reactivos": reactivos}
            for c in media_cols:
                st.session_state[f"p_th_{c}"] = tuple(rangos_r[c]["theta"])
                st.session_state[f"p_al_{c}"] = tuple(rangos_r[c]["alpha"])
                st.session_state[f"p_th0_{c}"] = float(inicios_r[c]["theta"])
                st.session_state[f"p_al0_{c}"] = float(inicios_r[c]["alpha"])
            st.session_state["p_four"] = int(esp["armonicos"])
            st.session_state["p_cuad"] = bool(esp["tendencia_cuadratica"])
            st.rerun()

        R0 = st.session_state["pitch_reco"]
        if R0:
            st.success(f"Estructura sugerida: {R0['esp']['armonicos']} armónico(s) de estacionalidad, "
                       f"tendencia cuadrática {'sí' if R0['esp']['tendencia_cuadratica'] else 'no'}. "
                       f"Solo con eso el R² llega a {R0['esp']['r2_estructura']:.2f}. "
                       "Rangos y puntos iniciales ya quedaron cargados abajo.")
            if R0.get("reactivos"):
                st.warning(
                    "**Inversión reactiva detectada en: " + ", ".join(R0["reactivos"]) + "**  \n"
                    "El presupuesto de esos canales sube cuando el KPI cae, así que su correlación "
                    "cruda es negativa aunque el efecto real sea positivo. Un modelo sin priors no "
                    "puede desenredar eso: o los anclas con benchmark, o los tratas como control.")
            _tabla(st, R0["tabla"])

    # ---------------- especificación ----------------
    with st.expander("🎛️ Especificación del modelo", expanded=True):
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown("**Estructura temporal**")
            usar_tend = st.checkbox("Tendencia lineal", value=True, key="p_tend")
            usar_cuad = st.checkbox("Tendencia cuadrática", key="p_cuad")
            n_fourier = st.slider("Armónicos de estacionalidad", 0, 6, key="p_four")
            periodo = st.number_input("Periodo estacional", 2.0, 400.0, step=0.01, key="p_periodo")
        with e2:
            st.markdown("**Transformaciones**")
            tipo_sat = st.radio("Curva de saturación", ["potencia", "hill"], horizontal=True,
                                key="p_sat")
            L = st.slider("Memoria máxima del adstock (periodos)", 2, 24, 12, key="p_L")
        with e3:
            st.markdown("**Búsqueda y validación**")
            objetivo = st.selectbox("Qué optimizar",
                                    ["Mixto: R² ajustado + fuera de muestra (recomendado)",
                                     "R² ajustado (dentro de muestra)",
                                     "R² fuera de muestra"], key="p_obj")
            n_iter = st.slider("Iteraciones de búsqueda", 50, 2000, 400, 50, key="p_iter")
            holdout = st.slider("Tamaño de cada ventana de validación", 0, max(int(n * 0.3), 1),
                                min(12, max(int(n * 0.15), 1)), key="p_hold")
            pliegues = st.slider("Ventanas de validación (origen móvil)", 1, 5, 3, key="p_plieg")
            refinar = st.checkbox("Refinamiento local (afina el R²)", value=True, key="p_ref")
            semilla = st.number_input("Semilla", 0, 10_000, 42, 1, key="p_seed")

        b1, b2, b3 = st.columns(3)
        umbral_r2 = b1.slider("R² mínimo aceptable (alerta)", 0.0, 0.95, 0.50, 0.05, key="p_umb")
        banda = (b2.number_input("Contribución mínima creíble %", 0.0, 100.0, 5.0, 1.0, key="p_bmin"),
                 b3.number_input("Contribución máxima creíble %", 0.0, 100.0, 40.0, 1.0, key="p_bmax"))

        st.markdown("**Rangos y puntos iniciales por canal**")
        st.caption("Canal · rango theta (adstock) · rango alpha (saturación) · θ inicial · α inicial. "
                   "El punto inicial se evalúa siempre y es desde donde arranca el refinamiento.")
        rangos, inicio_th, inicio_al = {}, [], []
        for c in media_cols:
            etiqueta = f"`{c}`" + (f" · {tipos.get(c)}" if tipos and c in tipos else "")
            r1, r2, r3, r4, r5 = st.columns([2.4, 2.6, 2.6, 1.2, 1.2])
            r1.markdown(etiqueta)
            th = r2.slider(f"rango theta {c}", 0.0, 0.95, step=0.05,
                           key=f"p_th_{c}", label_visibility="collapsed")
            al = r3.slider(f"rango alpha {c}", 0.1, 1.0, step=0.05,
                           key=f"p_al_{c}", label_visibility="collapsed")
            th0 = r4.number_input(f"th0 {c}", 0.0, 0.95, step=0.05,
                                  key=f"p_th0_{c}", label_visibility="collapsed")
            al0 = r5.number_input(f"al0 {c}", 0.05, 1.0, step=0.05,
                                  key=f"p_al0_{c}", label_visibility="collapsed")
            rangos[c] = {"theta": th, "alpha": al}
            inicio_th.append(th0)
            inicio_al.append(al0)

    # ---------------- benchmark ----------------
    with st.expander("🎯 Benchmark de contribución (opcional)", expanded=False):
        st.caption("Empuja la solución hacia contribuciones conocidas de un MMM previo o del "
                   "benchmark de categoría. Es una penalización blanda: si el ajuste se degrada mucho "
                   "al forzarla, los datos están discrepando del benchmark y eso ya es un hallazgo.")
        peso_b = st.slider("Peso del benchmark (0 = solo R²)", 0.0, 3.0, 0.0, 0.1, key="p_pesob")
        usar_total = st.checkbox("Anclar la contribución TOTAL de medios", key="p_utot")
        bench_total = st.number_input("Contribución total objetivo %", 0.0, 100.0, 20.0, 0.5,
                                      key="p_btot") if usar_total else None
        usar_canal = st.checkbox("Anclar por canal", key="p_ucan")
        bench_canal = {}
        if usar_canal:
            cols = st.columns(min(len(media_cols), 4))
            for j, c in enumerate(media_cols):
                with cols[j % len(cols)]:
                    v = st.number_input(f"{c} %", 0.0, 100.0, 0.0, 0.1, key=f"p_bch_{c}")
                    if v > 0:
                        bench_canal[j] = v
        anclar = st.slider("Buscar solo alrededor de los puntos iniciales (radio; 0 = libre)",
                           0.0, 0.5, 0.0, 0.05, key="p_anclar")

    # ---------------- dummies ----------------
    with st.expander(f"🏷️ Dummies aplicadas ({len(st.session_state['pitch_dummies'])})",
                     expanded=False):
        binarias = [c for c in d.columns
                    if c not in media_cols + [kpi_col, date_col]
                    and pd.api.types.is_numeric_dtype(d[c])
                    and set(pd.unique(d[c].dropna())) <= {0, 1}]
        if binarias:
            st.caption("Columnas 0/1 ya presentes en tu base y usadas como control: "
                       + ", ".join(binarias))
        c1, c2, c3 = st.columns(3)
        with c1:
            f_punto = st.date_input("Fecha puntual", value=None, key="p_dpunto")
            if st.button("Agregar fecha puntual", key="p_bpunto") and f_punto:
                st.session_state["pitch_dummies"].append(
                    {"tipo": "punto", "fechas": [str(f_punto)],
                     "nombre": f"D_pico_{f_punto:%Y%m%d}", "motivo": "manual"})
                st.rerun()
        with c2:
            f_ini = st.date_input("Inicio de periodo", value=None, key="p_dini")
            f_fin = st.date_input("Fin de periodo", value=None, key="p_dfin")
            if st.button("Agregar periodo", key="p_bper") and f_ini and f_fin:
                st.session_state["pitch_dummies"].append(
                    {"tipo": "rango", "inicio": str(f_ini), "fin": str(f_fin),
                     "nombre": f"D_periodo_{f_ini:%Y%m%d}", "motivo": "manual"})
                st.rerun()
        with c3:
            f_esc = st.date_input("Escalón desde", value=None, key="p_desc")
            mes_sel = st.multiselect("Dummies de mes", list(range(1, 13)), key="p_dmes")
            cb1, cb2 = st.columns(2)
            if cb1.button("Agregar escalón", key="p_besc") and f_esc:
                st.session_state["pitch_dummies"].append(
                    {"tipo": "escalon", "desde": str(f_esc),
                     "nombre": f"D_escalon_{f_esc:%Y%m%d}", "motivo": "manual"})
                st.rerun()
            if cb2.button("Agregar meses", key="p_bmes") and mes_sel:
                for m in mes_sel:
                    st.session_state["pitch_dummies"].append(
                        {"tipo": "mes", "mes": int(m), "nombre": f"D_mes_{int(m):02d}",
                         "motivo": "manual"})
                st.rerun()
        if st.session_state["pitch_dummies"]:
            st.dataframe(pd.DataFrame([{"dummy": x["nombre"], "tipo": x["tipo"],
                                        "motivo": x.get("motivo", "manual")}
                                       for x in st.session_state["pitch_dummies"]]))
            if st.button("🗑️ Quitar todas las dummies", key="p_bclear"):
                st.session_state["pitch_dummies"] = []
                st.rerun()

    if st.button("⚡ Estimar modelo", type="primary", key="p_run"):
        C_gen, nom_gen = construir_controles(n, usar_tend, usar_cuad, n_fourier, periodo)
        C_user = d[ctrl_cols].values.astype(float) if ctrl_cols else np.zeros((n, 0))
        C_dum, nom_dum = construir_dummies(fechas, st.session_state["pitch_dummies"])
        partes = [x for x in (C_user, C_gen, C_dum) if x.size]
        C = np.hstack(partes) if partes else np.zeros((n, 0))
        nombres_ctrl = list(ctrl_cols) + nom_gen + nom_dum

        bus = Buscador(mat_inv, C, y, tipo_sat, L, holdout, pliegues)
        bench = {"total": bench_total, "canal": bench_canal, "peso": peso_b}
        inicios = [{"thetas": inicio_th, "alphas": inicio_al}]
        barra = st.progress(0.0, text="Explorando combinaciones…")
        mejor, top = buscar(bus, rangos, n_iter, objetivo, banda, int(semilla), refinar,
                            bench=bench, inicios=inicios,
                            anclar=(anclar if anclar > 0 else None),
                            progreso=lambda p: barra.progress(min(p, 1.0),
                                                              text="Explorando combinaciones…"))
        barra.empty()
        st.session_state["pitch_res"] = dict(
            mejor=mejor, top=top, bus=bus, medios=media_cols, nombres_ctrl=nombres_ctrl,
            y=y, mat_inv=mat_inv, fechas=fechas, banda=banda, umbral_r2=umbral_r2,
            bench_canal=bench_canal,
            config={"date_col": date_col, "kpi_col": kpi_col, "media_cols": media_cols,
                    "control_cols": ctrl_cols, "saturacion": tipo_sat, "L_adstock": L,
                    "fourier": n_fourier, "periodo": periodo, "holdout": holdout,
                    "pliegues_validacion": pliegues, "objetivo": objetivo,
                    "iteraciones": n_iter, "semilla": int(semilla),
                    "benchmark": {"total": bench_total,
                                  "canal": {media_cols[j]: v for j, v in bench_canal.items()},
                                  "peso": peso_b, "anclaje": anclar},
                    "puntos_iniciales": {c: {"theta": inicio_th[j], "alpha": inicio_al[j]}
                                         for j, c in enumerate(media_cols)}})

    if "pitch_res" not in st.session_state:
        st.info("Configura la especificación y pulsa **Estimar modelo**. "
                "Si es tu primera corrida, pasa antes por el recomendador.")
        return

    R = st.session_state["pitch_res"]
    mejor, bus, medios, yv = R["mejor"], R["bus"], R["medios"], R["y"]
    if len(yv) != n or medios != media_cols:
        st.warning("Cambiaste el rango de fechas o la clasificación de variables desde la última "
                   "corrida. Vuelve a pulsar **Estimar modelo** para actualizar los resultados.")
    aportes, base, yhat, X = descomponer(bus, mejor, medios, R["nombres_ctrl"], yv)
    tabla = tabla_canales(medios, aportes, R["mat_inv"], yv, mejor, None, R["bench_canal"])
    avisos = diagnosticar(mejor, medios, tabla, R["banda"], R["umbral_r2"],
                          len(st.session_state["pitch_dummies"]))

    s_fit, s_desc, s_dum, s_curv, s_cand, s_exp = st.tabs(
        ["📊 Ajuste", "🥧 Descomposición", "🏷️ Dummies", "📈 Curvas", "🔎 Candidatos", "🧩 Export"])

    with s_fit:
        m = st.columns(5)
        m[0].metric("R²", f"{mejor['R2']:.3f}",
                    delta=f"{(mejor['R2']-R['umbral_r2'])*100:+.1f} pp vs mínimo")
        m[1].metric("R² ajustado", f"{mejor['R2_adj']:.3f}")
        m[2].metric("R² fuera de muestra",
                    "—" if not np.isfinite(mejor["R2_out"]) else f"{mejor['R2_out']:.3f}")
        m[3].metric("MAPE", f"{mejor['MAPE']:.1f}%")
        m[4].metric("Durbin-Watson", f"{mejor['DW']:.2f}")
        for a in avisos:
            getattr(st, a["sev"])(f"**{a['tema']}**  \n{a['detalle']}  \n➜ {a['sugerencia']}")
        f = fig_ajuste(R["fechas"], yv, yhat); st.pyplot(f); plt.close(f)

    with s_desc:
        c1, c2 = st.columns(2)
        c1.metric("Contribución total de medios", f"{mejor['pct_medios']:.1f}%")
        c2.metric("Base (tendencia + controles)", f"{100 - mejor['pct_medios']:.1f}%")
        _tabla(st, tabla)
        f = fig_barras_contrib(tabla); st.pyplot(f); plt.close(f)
        f = fig_area(R["fechas"], base, aportes); st.pyplot(f); plt.close(f)
        k = len(medios)
        st.markdown("**Controles, estructura temporal y dummies**")
        _tabla(st, pd.DataFrame({"variable": R["nombres_ctrl"],
                                   "coeficiente": np.round(mejor["beta"][k:], 3)}))

    with s_dum:
        st.markdown("**Dummies sugeridas por los residuales**")
        st.caption("Cada candidata se prueba en el modelo actual. `ΔR2_fuera` dice si explica algo o "
                   "solo absorbe una observación incómoda; `Δmedios_pp` dice cuánto desplaza la "
                   "contribución total de medios.")
        resid = yv - yhat
        specs = sugerir_dummies(R["fechas"], resid)
        tabla_d, specs_ord = evaluar_sugerencias(R["fechas"], X, yv, mejor["lam"], bus.pos_mask,
                                                 specs, bus.h, bus.pliegues, bus.k)
        if len(tabla_d) == 0:
            st.success("No hay residuales atípicos que justifiquen una dummy. El modelo está limpio.")
        else:
            z = np.abs(resid - np.median(resid))
            f = fig_residuales(R["fechas"], yv, yhat, np.argsort(-z)[:6]); st.pyplot(f); plt.close(f)
            _tabla(st, tabla_d)
            recomendadas = tabla_d[(tabla_d["ΔR2_adj"] > 0.005)
                                   & (tabla_d["veredicto"] == "explica")]["dummy"].tolist()[:4]
            elegidas = st.multiselect("Dummies a aplicar", tabla_d["dummy"].tolist(),
                                      default=recomendadas, key="p_elige_dum")
            if st.button("➕ Aplicar seleccionadas", key="p_aplica_dum"):
                actuales = {x["nombre"] for x in st.session_state["pitch_dummies"]}
                for nom, sp in zip(tabla_d["dummy"], specs_ord):
                    if nom in elegidas and nom not in actuales:
                        st.session_state["pitch_dummies"].append(sp)
                st.rerun()
            st.info("Después de aplicarlas, vuelve arriba y pulsa **Estimar modelo** para reajustar "
                    "los hiperparámetros con las dummies dentro. Si aplicas varias, ancla la "
                    "descomposición con el benchmark: las dummies tienden a inflar la contribución "
                    "de medios.")

    with s_curv:
        f = fig_curvas(bus, mejor, medios, R["mat_inv"]); st.pyplot(f); plt.close(f)
        filas = []
        for j, c in enumerate(medios):
            v = curva_respuesta(bus, mejor, j, np.array([1.0, 1.1]))
            inv = R["mat_inv"][:, j].sum()
            filas.append({"canal": c, "aporte_actual": round(v[0], 0),
                          "aporte_con_+10%": round(v[1], 0),
                          "KPI_marginal_por_1000": round((v[1] - v[0]) / (inv * 0.1) * 1000, 3)
                          if inv > 0 else np.nan})
        _tabla(st, pd.DataFrame(filas).sort_values("KPI_marginal_por_1000", ascending=False))

    with s_cand:
        st.caption("Los 15 mejores modelos. Si varios tienen R² parecido pero descomposiciones muy "
                   "distintas, esa dispersión es la incertidumbre real que el R² no muestra.")
        _tabla(st, pd.DataFrame([{
            "origen": c.get("origen", ""), "R2": round(c["R2"], 4), "R2_adj": round(c["R2_adj"], 4),
            "R2_out": None if not np.isfinite(c["R2_out"]) else round(c["R2_out"], 4),
            "MAPE": round(c["MAPE"], 2), "medios_%": round(c["pct_medios"], 1),
            "lambda": f"{c['lam']:.2e}", "canales_en_cero": c["n_cero"],
            "score": round(c["score"], 4)} for c in R["top"]]))

    with s_exp:
        meta = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "filas": len(yv),
                "medios": len(medios),
                "periodo": f"{pd.Timestamp(R['fechas'].iloc[0]):%Y-%m-%d} a "
                           f"{pd.Timestamp(R['fechas'].iloc[-1]):%Y-%m-%d}"}
        figs_b64 = [("KPI real vs estimado", fig_to_b64(fig_ajuste(R["fechas"], yv, yhat))),
                    ("Descomposición del KPI", fig_to_b64(fig_area(R["fechas"], base, aportes))),
                    ("Contribución por canal", fig_to_b64(fig_barras_contrib(tabla)))]
        html = informe_html(CLIENTE, meta, mejor, tabla, avisos, figs_b64,
                            st.session_state["pitch_dummies"])
        st.download_button("⬇️ Descargar informe (HTML)", html,
                           file_name=f"pitch_mmm_{datetime.now():%Y%m%d}.html", mime="text/html",
                           key="p_dl_html")
        codigo = export_json(R["config"], mejor, medios, tabla, st.session_state["pitch_dummies"])
        st.code(codigo, language="python")
        st.download_button("⬇️ Descargar config_pitch.py", codigo, file_name="config_pitch.py",
                           mime="text/x-python", key="p_dl_py")
        st.download_button("⬇️ Tabla de canales (CSV)", tabla.to_csv(index=False),
                           file_name="canales_pitch.csv", mime="text/csv", key="p_dl_csv")
