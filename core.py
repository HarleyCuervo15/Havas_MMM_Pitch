"""
core.py — Lógica de datos, estadísticos y temas de color.
App de Preproceso + EDA para MMM (Meridian) · CSA
Separado de la interfaz para poder probarlo sin Streamlit.
"""
import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

# ============================================================================
# TEMAS DE COLOR
# ============================================================================
TEMAS = {
    "Tigo – Millicom": {"primario": "#00263A", "secundario": "#0088CE", "acento": "#E4002B"},
    "CSA (rojo)":      {"primario": "#1A1A1A", "secundario": "#5A5A5A", "acento": "#E4002B"},
    "Océano":          {"primario": "#0B3954", "secundario": "#087E8B", "acento": "#FF5A5F"},
    "Bosque":          {"primario": "#1B3A2F", "secundario": "#2E8B57", "acento": "#E9A700"},
    "Violeta":         {"primario": "#2D1B4E", "secundario": "#7B4FBF", "acento": "#FF7A00"},
    "Grafito":         {"primario": "#22272B", "secundario": "#4E5D6C", "acento": "#00A6A6"},
}
TEMA_DEFECTO = "Tigo – Millicom"

# Grupos de variables. 'control' indica si entran como control en Meridian.
GRUPOS = {
    "medios_online":     {"label": "Medios ONLINE (inversión)",      "control": False},
    "medios_offline":    {"label": "Medios OFFLINE (inversión)",     "control": False},
    "competencia_inv":   {"label": "Inversión de COMPETENCIA",       "control": True},
    "competencia_share": {"label": "SHARE de competencia (SOV)",     "control": True},
    "negocio":           {"label": "Variables de NEGOCIO",           "control": True},
    "macro":             {"label": "Variables MACROECONÓMICAS",      "control": True},
}


def paleta_medios(p, n):
    """Degradado del color secundario para n series de medios."""
    base = sns.light_palette(p["secundario"], n_colors=max(n + 2, 3), reverse=True)
    return list(base)[:n] if n else []


def paleta_dual(p):
    """Colores para Online vs Offline."""
    return {"Online": p["secundario"], "Offline": p["acento"]}


# ============================================================================
# DATOS
# ============================================================================
def leer_datos(contenido: bytes, nombre: str) -> pd.DataFrame:
    buf = io.BytesIO(contenido)
    if nombre.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    return pd.read_csv(buf)


def _racha_max_ceros(s):
    best = cur = 0
    for v in (s == 0).astype(int).values:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _picos(s, f):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    n = int(((s < q1 - f * iqr) | (s > q3 + f * iqr)).sum())
    return n, round(n / len(s) * 100, 1)


def _ccf_best(x, y, max_lag=8):
    best_lag, best_r = 0, 0.0
    for L in range(0, max_lag + 1):
        r = x.shift(L).corr(y, method="spearman")
        if pd.notna(r) and abs(r) > abs(best_r):
            best_r, best_lag = r, L
    return best_lag, round(best_r, 3)


# ============================================================================
# TABLERO Y ALERTAS
# ============================================================================
def construir_tablero(df, media, kpi_col, iqr_factor, tipos=None):
    """tipos: dict canal -> 'Online'/'Offline'."""
    y = df[kpi_col]
    mi = mutual_info_regression(
        df[media].fillna(df[media].median()).values, y.values, random_state=0)
    total = df[media].sum().sum()
    soi = (df[media].sum() / total * 100).round(1) if total else df[media].sum() * 0
    filas = []
    for i, c in enumerate(media):
        npk, _ = _picos(df[c], iqr_factor)
        blag, bcorr = _ccf_best(df[c], y)
        filas.append({
            "canal": c,
            "tipo": (tipos or {}).get(c, "—"),
            "SOI_%": float(soi[c]),
            "%ceros": round((df[c] == 0).mean() * 100, 1),
            "racha_ceros": _racha_max_ceros(df[c]),
            "picos": npk,
            "corr_KPI": round(df[c].corr(y, method="spearman"), 3),
            "MI_KPI": round(float(mi[i]), 3),
            "mejor_lag": blag,
            "corr_en_lag": bcorr,
        })
    return pd.DataFrame(filas)


def generar_alertas(tablero, n_filas, U):
    alertas = []

    def a(sev, cat, var, msg, fix):
        alertas.append({"sev": sev, "tema": cat, "variable": var,
                        "detalle": msg, "sugerencia": fix})

    for _, r in tablero.iterrows():
        v = r["canal"]
        if r["SOI_%"] < U["soi_minimo_pct"]:
            a("warning", "SOI bajo", v,
              f"Pesa solo {r['SOI_%']}% de la inversión total (< {U['soi_minimo_pct']}%).",
              "Agrúpalo con un canal parecido o valida si aporta señal suficiente.")
        if r["%ceros"] > U["pct_ceros"]:
            a("warning", "Muchos silencios", v,
              f"El {r['%ceros']}% de las semanas está en cero.",
              "Canal intermitente: su adstock será inestable. Considera agruparlo.")
        if r["racha_ceros"] >= U["racha_ceros"]:
            a("error", "Apagón prolongado", v,
              f"Estuvo {int(r['racha_ceros'])} semanas seguidas en cero.",
              "Confirma si fue pausa real de campaña o datos faltantes.")
        if (r["picos"] / n_filas * 100) > U["pct_picos"]:
            a("warning", "Picos atípicos", v,
              f"Tiene {int(r['picos'])} semanas con valores muy fuera de rango.",
              "Revisa esas semanas: suelen ser errores de carga o promociones.")
        debil = abs(r["corr_KPI"]) < U["corr_kpi_minima"]
        fuerte_lag = abs(r["corr_en_lag"]) >= U["corr_kpi_minima"]
        if debil and fuerte_lag:
            a("info", "Efecto con rezago", v,
              f"Casi no correlaciona hoy, pero sí {r['corr_en_lag']} con "
              f"{int(r['mejor_lag'])} semanas de rezago.",
              "Normal en TV/branding. Súbele el adstock (alpha_m) en la Parte 4.")
        elif debil:
            a("warning", "Relación débil", v,
              f"Correlación con el KPI muy baja ({r['corr_KPI']}) incluso con rezago.",
              "Puede aportar poco. Revisa si vale la pena mantenerlo aparte.")
    return alertas


def alertas_mix(tablero, U):
    """Alertas sobre el balance Online/Offline."""
    out = []
    if "tipo" not in tablero.columns:
        return out
    g = tablero.groupby("tipo")["SOI_%"].sum()
    for t in ("Online", "Offline"):
        if t in g.index and g[t] < U.get("mix_minimo_pct", 15):
            out.append({"sev": "info", "tema": "Mix desbalanceado", "variable": t,
                        "detalle": f"Los medios {t} concentran solo {g[t]:.1f}% de la inversión.",
                        "sugerencia": "Con poca inversión relativa, su efecto será difícil de "
                                      "estimar; considera agrupar esos canales."})
    return out


# ============================================================================
# ESTADÍSTICOS
# ============================================================================
def vif_table(df, cols):
    X = df[cols].fillna(df[cols].median())
    Xs = StandardScaler().fit_transform(X)
    out = []
    for i, c in enumerate(cols):
        y = Xs[:, i]
        Xo = np.delete(Xs, i, axis=1)
        r2 = LinearRegression().fit(Xo, y).score(Xo, y) if Xo.shape[1] else 0.0
        out.append({"variable": c, "VIF": round(1 / (1 - r2), 2) if r2 < 1 else np.inf})
    return pd.DataFrame(out).sort_values("VIF", ascending=False).reset_index(drop=True)


def descomposicion_varianza(df, kpi_col, media, date_col):
    d = df.reset_index(drop=True)
    t = np.arange(len(d))
    mes = pd.to_datetime(d[date_col]).dt.month
    X_ts = np.column_stack([t, t ** 2, pd.get_dummies(mes, drop_first=True).values]).astype(float)
    y = d[kpi_col].values.astype(float)
    resid = y - LinearRegression().fit(X_ts, y).predict(X_ts)
    Xm = d[media].fillna(d[media].median()).values
    r2m = LinearRegression().fit(Xm, resid).score(Xm, resid)
    var_total, var_resid = np.var(y), np.var(resid)
    return (round((1 - var_resid / var_total) * 100, 1),
            round(r2m * 100, 1),
            round(max(r2m, 0) * (var_resid / var_total) * 100, 1))


def eficiencia(tablero):
    cols = ["canal", "tipo", "SOI_%", "corr_KPI", "MI_KPI", "mejor_lag"]
    ef = tablero[[c for c in cols if c in tablero.columns]].copy()
    ef["fuerza_senal"] = ef["corr_KPI"].abs()
    ef["rank_SOI"] = ef["SOI_%"].rank(ascending=False)
    ef["rank_senal"] = ef["fuerza_senal"].rank(ascending=False)
    ef["lectura"] = np.where(ef["rank_senal"] < ef["rank_SOI"], "🟢 aporta más de lo que gasta",
                     np.where(ef["rank_senal"] > ef["rank_SOI"], "🔴 gasta más de lo que aporta",
                              "≈ equilibrado"))
    return ef.sort_values("fuerza_senal", ascending=False)


def lectura_curvas(df, media, kpi_col, tipos=None):
    y = df[kpi_col]
    filas, overrides = [], {}
    for c in media:
        blag, bcorr = _ccf_best(df[c], y)
        cv = round(df[c].std() / df[c].mean(), 2) if df[c].mean() > 0 else np.nan
        if blag <= 1:
            fam, maxlag, lo, hi = "geométrico (rápido)", "2 a 8", 0.1, 0.5
        else:
            fam, maxlag, lo, hi = "binomial (pico retardado)", "4 a 20", 0.4, 0.9
        nota = "rango estrecho: apóyate en el prior" if (pd.notna(cv) and cv < 0.4) else "rango ok"
        filas.append({"canal": c, "tipo": (tipos or {}).get(c, "—"), "mejor_lag": blag,
                      "corr_en_lag": bcorr, "adstock_sugerido": fam,
                      "max_lag_sugerido": maxlag, "alpha_m_sugerido": f"{lo} - {hi}",
                      "saturación": "cóncava (Hill)", "rango_gasto": nota})
        overrides[c] = {"adstock_low": lo, "adstock_high": hi}
    return pd.DataFrame(filas), overrides


def densidad_senal(n_obs, n_media, n_control, n_knots):
    params = n_media * 3 + n_control + 1 + n_knots
    return round(n_obs / params, 1), params


def ccf_matriz(df, variables, kpi_col, max_lag=8):
    y = df[kpi_col]
    data = {f"lag {L}": [] for L in range(max_lag + 1)}
    idx = []
    for c in variables:
        idx.append(c)
        for L in range(max_lag + 1):
            r = df[c].shift(L).corr(y, method="spearman")
            data[f"lag {L}"].append(round(r, 3) if pd.notna(r) else np.nan)
    return pd.DataFrame(data, index=idx)


def control_vs_kpi(df, ctrl, kpi_col, grupo_de=None):
    y = df[kpi_col]
    mi = mutual_info_regression(df[ctrl].fillna(df[ctrl].median()).values,
                                y.values, random_state=0)
    rows = []
    for i, c in enumerate(ctrl):
        blag, bcorr = _ccf_best(df[c], y)
        rows.append({"variable": c,
                     "grupo": (grupo_de or {}).get(c, "—"),
                     "corr_KPI": round(df[c].corr(y, method="spearman"), 3),
                     "MI_KPI": round(float(mi[i]), 3),
                     "mejor_lag": blag, "corr_en_lag": bcorr})
    return pd.DataFrame(rows)


# ============================================================================
# SHARES
# ============================================================================
def tabla_share(df, media, tipos=None):
    """Share of Investment por canal y por tipo."""
    tot = df[media].sum().sum()
    s = (df[media].sum() / tot * 100) if tot else df[media].sum() * 0
    t = pd.DataFrame({"canal": media, "SOI_%": s.values.round(2)})
    if tipos:
        t["tipo"] = [tipos.get(c, "—") for c in media]
    return t.sort_values("SOI_%", ascending=False).reset_index(drop=True)


def share_online_offline(df, media, tipos):
    on = [c for c in media if tipos.get(c) == "Online"]
    off = [c for c in media if tipos.get(c) == "Offline"]
    tot = df[media].sum().sum()
    if not tot:
        return {"Online": 0.0, "Offline": 0.0}
    return {"Online": round(df[on].sum().sum() / tot * 100, 1) if on else 0.0,
            "Offline": round(df[off].sum().sum() / tot * 100, 1) if off else 0.0}


def share_of_spend(df, media, competencia_inv, date_col, freq="MS"):
    """Nuestro share frente a la inversión de competencia, en el tiempo."""
    if not competencia_inv:
        return None
    d = df[[date_col] + media + competencia_inv].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    g = d.set_index(date_col).resample(freq).sum()
    nuestro = g[media].sum(axis=1)
    comp = g[competencia_inv].sum(axis=1)
    tot = (nuestro + comp).replace(0, np.nan)
    return pd.DataFrame({"Nosotros_%": (nuestro / tot * 100).round(1),
                         "Competencia_%": (comp / tot * 100).round(1)}).dropna()


# ============================================================================
# GRÁFICAS
# ============================================================================
def _fin(fig):
    fig.tight_layout()
    return fig


def fig_inversion_apilada(df, media, date_col, p, freq="MS", tipos=None):
    d = df[[date_col] + media].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    g = d.set_index(date_col)[media].resample(freq).sum()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    bottom = np.zeros(len(g))
    colores = paleta_medios(p, len(media))
    ancho = {"W": 5, "MS": 22, "QS": 70}.get(freq, 22)
    for c, col in zip(media, colores):
        ax.bar(g.index, g[c].values, bottom=bottom, width=ancho, label=c, color=col)
        bottom += g[c].values
    ax.set_title("Inversión por medio a lo largo del tiempo", color=p["primario"], fontweight="bold")
    ax.set_ylabel("Inversión")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    return _fin(fig)


def fig_kpi_trend(df, kpi_col, date_col, p):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(d[date_col], d[kpi_col], color=p["primario"], lw=2)
    ax.fill_between(d[date_col], d[kpi_col], alpha=0.08, color=p["primario"])
    ax.set_title(f"Tendencia del KPI: {kpi_col}", color=p["primario"], fontweight="bold")
    return _fin(fig)


def fig_overlay(df, media, kpi_col, date_col, p):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)
    total = d[media].sum(axis=1)
    fig, ax1 = plt.subplots(figsize=(11, 4.2))
    ax1.bar(d[date_col], total, width=5, alpha=0.35, color=p["secundario"])
    ax1.set_ylabel("Inversión total", color=p["secundario"])
    ax2 = ax1.twinx()
    ax2.plot(d[date_col], d[kpi_col], color=p["acento"], lw=2)
    ax2.set_ylabel("KPI", color=p["acento"])
    ax1.set_title("Inversión total vs KPI — ¿se mueven juntos?", color=p["primario"], fontweight="bold")
    return _fin(fig)


def fig_share_donut(tabla, p):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    colores = paleta_medios(p, len(tabla))
    ax.pie(tabla["SOI_%"], labels=tabla["canal"], autopct="%1.1f%%",
           colors=colores, startangle=90, wedgeprops={"width": 0.42, "edgecolor": "white"},
           textprops={"fontsize": 9})
    ax.set_title("Share of Investment por canal", color=p["primario"], fontweight="bold")
    return _fin(fig)


def fig_share_online_offline(shares, p):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    cols = paleta_dual(p)
    labels = [k for k in ("Online", "Offline") if shares.get(k, 0) > 0]
    vals = [shares[k] for k in labels]
    ax.pie(vals, labels=labels, autopct="%1.1f%%",
           colors=[cols[k] for k in labels], startangle=90,
           wedgeprops={"width": 0.42, "edgecolor": "white"}, textprops={"fontsize": 11})
    ax.set_title("Mix Online vs Offline", color=p["primario"], fontweight="bold")
    return _fin(fig)


def fig_share_temporal(df, media, date_col, p, freq="MS"):
    """Share 100% apilado en el tiempo."""
    d = df[[date_col] + media].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    g = d.set_index(date_col)[media].resample(freq).sum()
    tot = g.sum(axis=1).replace(0, np.nan)
    pct = g.div(tot, axis=0).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.stackplot(pct.index, [pct[c].values for c in media], labels=media,
                 colors=paleta_medios(p, len(media)))
    ax.set_ylim(0, 100)
    ax.set_ylabel("% del total")
    ax.set_title("Evolución del share de inversión por canal", color=p["primario"], fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    return _fin(fig)


def fig_share_of_spend(sos, p):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(sos.index, sos["Nosotros_%"].values, sos["Competencia_%"].values,
                 labels=["Nosotros", "Competencia"],
                 colors=[p["secundario"], p["acento"]], alpha=.85)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% de la inversión del mercado")
    ax.set_title("Share of Spend: nosotros vs competencia", color=p["primario"], fontweight="bold")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.28))
    return _fin(fig)


def fig_sov_lineas(df, cols, date_col, p, titulo):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for c, col in zip(cols, paleta_medios(p, len(cols))):
        ax.plot(d[date_col], d[c], lw=1.8, label=c, color=col)
    ax.set_title(titulo, color=p["primario"], fontweight="bold")
    ax.legend(fontsize=8, ncol=3)
    return _fin(fig)


def fig_ccf_heatmap(m, p):
    fig, ax = plt.subplots(figsize=(1.05 * m.shape[1] + 2, 0.55 * m.shape[0] + 2))
    sns.heatmap(m, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                cbar_kws={"label": "Spearman vs KPI"}, ax=ax)
    ax.set_title("Correlación cruzada con el KPI por rezago", color=p["primario"], fontweight="bold")
    ax.set_xlabel("Rezago (semanas)")
    ax.set_ylabel("")
    return _fin(fig)


def fig_tendencias_grupo(df, cols, date_col, p, titulo):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)
    n = len(cols)
    ncols = min(2, n) or 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.9 * nrows), squeeze=False)
    axes = axes.reshape(-1)
    for ax, c in zip(axes, cols):
        ax.plot(d[date_col], d[c], color=p["primario"], lw=1.7)
        ax.fill_between(d[date_col], d[c], alpha=0.07, color=p["primario"])
        ax.set_title(c, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(titulo, color=p["primario"], fontweight="bold", y=1.02)
    return _fin(fig)


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b


# ============================================================================
# EXPORTACIÓN
# ============================================================================
def generar_codigo_colab(config, umbrales, overrides, start=None, end=None):
    rango = ""
    if start and end:
        rango = ("\n# 4) Rango de fechas seleccionado (Parte 1.3 del notebook):\n"
                 f'START = "{start}"\nEND   = "{end}"\n')
    return ("# === Generado por la app de EDA (CSA) — pégalo en Colab ===\n"
            "# 1) Reemplaza el bloque CONFIG de la Parte 1.2-bis por este:\n"
            f"CONFIG = {json.dumps(config, ensure_ascii=False, indent=4)}\n\n"
            "# 2) Umbrales de EDA usados:\n"
            f"UMBRALES = {json.dumps(umbrales, ensure_ascii=False, indent=4)}\n\n"
            "# 3) Priors sugeridos por canal para HP['overrides'] (Parte 4):\n"
            f"HP_OVERRIDES = {json.dumps(overrides, ensure_ascii=False, indent=4)}\n"
            f"{rango}")


def generar_informe_html(cliente, meta, alertas, tablero, figs_b64, seccionB, p, grupos_resumen):
    color = {"error": p["acento"], "warning": "#E8A400", "info": p["secundario"]}
    etiqueta = {"error": "CRÍTICO", "warning": "REVISAR", "info": "INFO"}
    if alertas:
        orden = {"error": 0, "warning": 1, "info": 2}
        items = "".join(
            f'<div style="border-left:5px solid {color[a["sev"]]};background:#fff;'
            f'padding:10px 14px;margin:8px 0;border-radius:6px;">'
            f'<b style="color:{color[a["sev"]]}">{etiqueta[a["sev"]]} · {a["tema"]} — {a["variable"]}</b><br>'
            f'{a["detalle"]}<br><i style="color:#556;">➜ {a["sugerencia"]}</i></div>'
            for a in sorted(alertas, key=lambda x: orden.get(x["sev"], 3)))
    else:
        items = "<p>Sin alertas: los canales se ven sanos. ✅</p>"

    imgs = "".join(f'<h3>{t}</h3><img src="data:image/png;base64,{b}" '
                   f'style="width:100%;max-width:900px;">' for t, b in figs_b64)
    grupos_html = "".join(f"<li><b>{k}:</b> {v}</li>" for k, v in grupos_resumen.items() if v)
    pct_ts, pct_media, techo = seccionB["descomp"]
    ratio, params = seccionB["densidad"]

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Informe EDA MMM · {cliente}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;color:#1b2a33;margin:0;background:#f4f7f9;}}
.wrap{{max-width:960px;margin:0 auto;padding:24px;}}
.header{{background:linear-gradient(120deg,{p['primario']},{p['secundario']});color:#fff;
padding:26px 28px;border-radius:12px;}}
.kick{{color:{p['acento']};font-weight:800;letter-spacing:.14em;font-size:.75rem;text-transform:uppercase;
background:#fff;display:inline-block;padding:2px 10px;border-radius:12px;}}
h1{{margin:8px 0;font-size:1.7rem;}}
h2{{color:{p['primario']};border-bottom:2px solid #e0e8ee;padding-bottom:6px;margin-top:28px;}}
h3{{color:{p['primario']};}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;}}
th{{background:{p['primario']};color:#fff;padding:6px 8px;text-align:left;}}
td{{padding:5px 8px;border-bottom:1px solid #e6edf1;}}
.cards{{display:flex;gap:12px;margin:10px 0;}}
.card{{flex:1;background:#fff;border-left:5px solid {p['secundario']};border-radius:8px;padding:12px 14px;}}
.card b{{font-size:1.4rem;color:{p['primario']};}}
ul{{background:#fff;border-radius:8px;padding:14px 30px;}}
.foot{{color:#8aa0ad;font-size:.8rem;text-align:center;margin-top:26px;}}
</style></head><body><div class="wrap">
<div class="header"><span class="kick">Consumer Science &amp; Analytics · MMM</span>
<h1>Informe de EDA — {cliente}</h1>
<div>Generado el {meta['fecha']} · Periodo: {meta.get('periodo','—')} · {meta['filas']} semanas</div></div>

<h2>1 · Variables analizadas</h2><ul>{grupos_html}</ul>

<h2>2 · Alertas</h2>{items}

<h2>3 · Tendencias y shares</h2>{imgs}

<h2>4 · Tablero por canal</h2>{tablero.to_html(index=False, border=0)}

<h2>5 · Lectura para el modelador</h2>
<div class="cards">
<div class="card"><b>{pct_ts}%</b><br>Tendencia + estacionalidad</div>
<div class="card"><b>~{techo}%</b><br>Techo de contribución de medios</div>
<div class="card"><b>{ratio}</b><br>Obs. por parámetro</div>
</div>
<p style="color:#556;font-size:.85rem;">El techo de contribución es una referencia para los priors
de ROI, no la contribución final. La densidad indica cuánto pesarán los datos frente a los priors.</p>

<div class="foot">Consumer Science &amp; Analytics · Documento de trabajo para {cliente}</div>
</div></body></html>"""
