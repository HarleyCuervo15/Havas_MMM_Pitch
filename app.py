"""
app.py — Preproceso + EDA para Marketing Mix Modeling (Meridian)
Consumer Science & Analytics · Cliente: Tigo–Millicom
Ejecuta:  streamlit run app.py
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

import core
from core import TEMAS, TEMA_DEFECTO, GRUPOS

CLIENTE = "Tigo – Millicom"
LOGO_PATH = "logo.png"          # si lo guardas en assets/, cambia a "assets/logo.png"

st.set_page_config(page_title="EDA MMM · CSA", page_icon="📊", layout="wide")


# ============================================================================
# TEMA DE COLOR
# ============================================================================
def selector_tema():
    if "paleta" not in st.session_state:
        st.session_state.paleta = dict(TEMAS[TEMA_DEFECTO])
    with st.expander("🎨 Colores de la app", expanded=False):
        c1, c2 = st.columns([1, 2])
        with c1:
            nombre = st.selectbox("Paleta predefinida", list(TEMAS.keys()),
                                  index=list(TEMAS.keys()).index(TEMA_DEFECTO))
            if st.button("Aplicar paleta"):
                st.session_state.paleta = dict(TEMAS[nombre])
        with c2:
            st.caption("O ajusta cada color a mano:")
            k1, k2, k3 = st.columns(3)
            p = st.session_state.paleta
            p["primario"] = k1.color_picker("Primario", p["primario"])
            p["secundario"] = k2.color_picker("Secundario", p["secundario"])
            p["acento"] = k3.color_picker("Acento", p["acento"])
            st.session_state.paleta = p
    return st.session_state.paleta


def aplicar_css(p):
    st.markdown(f"""<style>
.block-container {{ padding-top: 2.2rem; }}
.csa-banner {{ background: linear-gradient(120deg, {p['primario']} 0%, {p['secundario']} 150%);
  border-radius: 14px; padding: 20px 26px; color:#fff; margin-bottom: 8px; }}
.csa-kicker {{ color:#fff; background:{p['acento']}; display:inline-block; padding:2px 10px;
  border-radius:12px; font-weight:800; letter-spacing:.12em; font-size:.72rem; text-transform:uppercase; }}
.csa-title {{ font-size:1.9rem; font-weight:800; line-height:1.1; margin:8px 0 4px 0; }}
.csa-sub {{ color:#eaf3f8; font-size:.95rem; }}
.csa-badge {{ display:inline-block; background:#ffffff22; border:1px solid #ffffff55; color:#fff;
  font-weight:700; padding:3px 12px; border-radius:20px; font-size:.8rem; margin-top:8px; }}
h2, h3 {{ color:{p['primario']}; }}
div[data-testid="stMetric"] {{ background:#F6F9FB; border:1px solid #e3edf3;
  border-left:5px solid {p['secundario']}; border-radius:10px; padding:12px 14px; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; }}
.csa-foot {{ color:#8aa0ad; font-size:.8rem; text-align:right; margin-top:10px; }}
</style>""", unsafe_allow_html=True)


def banner(p):
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(
            '<div class="csa-banner">'
            '<span class="csa-kicker">Consumer Science &amp; Analytics · MMM</span>'
            '<div class="csa-title">Preproceso + EDA para MMM</div>'
            '<div class="csa-sub">Carga tu base, clasifica las variables, revisa alertas y shares, '
            'y genera el código para Meridian en Colab.</div>'
            f'<div class="csa-badge">Cliente: {CLIENTE}</div></div>', unsafe_allow_html=True)
    with c2:
        st.write("")
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown('<div style="border:1px dashed #9bb3c1;border-radius:12px;padding:20px 8px;'
                        'text-align:center;color:#7f97a5;font-size:.8rem;">Coloca el logo en '
                        f'<code>{LOGO_PATH}</code></div>', unsafe_allow_html=True)


PALETA = selector_tema()
aplicar_css(PALETA)
banner(PALETA)

# ============================================================================
# CARGA
# ============================================================================
archivo = st.file_uploader("Sube tu base (.xlsx o .csv)", type=["xlsx", "xls", "csv"])
if archivo is None:
    st.info("👆 Sube un archivo para empezar. Debe tener una columna de fecha y el resto numéricas.")
    st.stop()

df = core.leer_datos(archivo.getvalue(), archivo.name)
numeric = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

# ============================================================================
# CONFIGURACIÓN DE CAMPOS
# ============================================================================
with st.expander("⚙️ Configuración de campos", expanded=True):
    st.markdown("**1 · Fecha y KPI**")
    c1, c2 = st.columns([1, 2])
    _g = next((c for c in all_cols if str(c).lower() in
               ("fecha", "date", "semana", "week", "periodo", "mes")), all_cols[0])
    date_col = c1.selectbox("Columna de fecha", all_cols, index=all_cols.index(_g))
    with c2:
        modo = st.radio("Cómo definir el KPI", ["Una sola columna", "Sumar varias columnas"],
                        index=0, horizontal=True)
        if modo == "Una sola columna":
            kpi_col = st.selectbox("Columna KPI", numeric)
            kpi_source, kpi_mode_key = [kpi_col], "single"
        else:
            kpi_col = st.text_input("Nombre del KPI", "KPI")
            kpi_source, kpi_mode_key = st.multiselect("Columnas a sumar", numeric), "combine"

    st.markdown("**2 · Clasificación de variables**")
    st.caption("Cada grupo se analiza por separado. Los cuatro últimos entran como *control* en Meridian.")
    usar_patron = st.checkbox("Usar patrones de texto para filtrar (útil con muchas columnas)")
    sel = {}
    filas = [("medios_online", "medios_offline"), ("competencia_inv", "competencia_share"),
             ("negocio", "macro")]
    usados = set(kpi_source)
    for izq, der in filas:
        ca, cb = st.columns(2)
        for col, key in ((ca, izq), (cb, der)):
            with col:
                disponibles = [c for c in numeric if c not in usados]
                if usar_patron:
                    pat = st.text_input(f"Patrón · {GRUPOS[key]['label']}", "", key=f"pat_{key}")
                    sel[key] = [c for c in disponibles if pat and pat.lower() in str(c).lower()]
                    st.caption(f"{len(sel[key])} columnas coinciden")
                else:
                    sel[key] = st.multiselect(GRUPOS[key]["label"], disponibles, key=f"ms_{key}")
                usados |= set(sel[key])

    with st.expander("Umbrales de alertas y densidad"):
        u1, u2, u3, u4 = st.columns(4)
        UMBRALES = {
            "soi_minimo_pct": u1.number_input("SOI mínimo %", 0.0, 100.0, 10.0, 1.0),
            "pct_ceros": u2.number_input("% ceros 'silencios'", 0.0, 100.0, 30.0, 5.0),
            "racha_ceros": int(u3.number_input("Racha ceros (sem)", 1, 52, 6, 1)),
            "iqr_factor": u4.number_input("Sensib. picos (IQR)", 0.5, 3.0, 1.5, 0.1),
            "pct_picos": u1.number_input("% picos alerta", 0.0, 100.0, 4.0, 1.0),
            "corr_kpi_minima": u2.number_input("Corr. mínima KPI", 0.0, 1.0, 0.15, 0.05),
            "vif_alerta": u3.number_input("VIF de alerta", 1.0, 50.0, 5.0, 1.0),
            "mix_minimo_pct": u4.number_input("Mix mínimo On/Off %", 0.0, 50.0, 15.0, 5.0),
        }
        n_knots = int(u1.number_input("Knots estimados", 2, 60, 12, 1))

# ---- Derivados ----
media_cols = sel["medios_online"] + sel["medios_offline"]
ctrl_cols = (sel["competencia_inv"] + sel["competencia_share"] + sel["negocio"] + sel["macro"])
tipos = {**{c: "Online" for c in sel["medios_online"]},
         **{c: "Offline" for c in sel["medios_offline"]}}
grupo_de = {c: k for k, v in sel.items() for c in v}

errores = []
if not kpi_source:
    errores.append("Elige al menos una columna para el KPI.")
if not media_cols:
    errores.append("Elige al menos un medio (online u offline).")
if errores:
    for e in errores:
        st.error("⚠️ " + e)
    st.stop()

# ---- df base + recorte de fechas ----
work = df.copy()
work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
if kpi_mode_key == "combine":
    work[kpi_col] = work[kpi_source].sum(axis=1)
df_full = (work[[date_col, kpi_col] + media_cols + ctrl_cols]
           .dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True))

st.markdown("#### 🗓️ Rango de fechas a revisar")
fmin, fmax = df_full[date_col].min().to_pydatetime(), df_full[date_col].max().to_pydatetime()
if fmin >= fmax:
    r0 = r1 = pd.Timestamp(fmin)
    st.caption("Solo hay una fecha en los datos.")
else:
    rango = st.slider("Arrastra los extremos para acotar el periodo", min_value=fmin,
                      max_value=fmax, value=(fmin, fmax), format="YYYY-MM-DD")
    r0, r1 = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])

df_model = df_full[(df_full[date_col] >= r0) & (df_full[date_col] <= r1)].reset_index(drop=True)
f1, f2, f3, f4 = st.columns(4)
f1.metric("Semanas", len(df_model))
f2.metric("Desde", r0.strftime("%Y-%m-%d"))
f3.metric("Hasta", r1.strftime("%Y-%m-%d"))
f4.metric("Medios", f"{len(sel['medios_online'])} on / {len(sel['medios_offline'])} off")

if len(df_model) < 5:
    st.error("El rango tiene muy pocos datos (< 5 semanas). Amplía el periodo.")
    st.stop()
if len(df_model) < 20:
    st.warning("Pocas semanas en el rango: algunos estadísticos pueden ser inestables.")

FECHA_INI, FECHA_FIN = r0.strftime("%Y-%m-%d"), r1.strftime("%Y-%m-%d")
predictores = media_cols + ctrl_cols
CONFIG = {"date_col": date_col, "kpi_mode": kpi_mode_key, "kpi_source_cols": kpi_source,
          "kpi_col": kpi_col, "media_cols": media_cols, "control_cols": ctrl_cols,
          "grupos": {k: v for k, v in sel.items()}}

# ---- cálculos compartidos ----
tablero = core.construir_tablero(df_model, media_cols, kpi_col, UMBRALES["iqr_factor"], tipos)
alertas = core.generar_alertas(tablero, len(df_model), UMBRALES) + core.alertas_mix(tablero, UMBRALES)
seccionB = {"descomp": core.descomposicion_varianza(df_model, kpi_col, media_cols, date_col),
            "densidad": core.densidad_senal(len(df_model), len(media_cols), len(ctrl_cols), n_knots)}
P = PALETA

t_datos, t_tend, t_share, t_alert, t_model, t_inf, t_code = st.tabs(
    ["📄 Datos", "📈 Tendencias", "🥧 Shares", "🚨 Alertas", "🎯 Modelador", "📑 Informe", "🧩 Código Colab"])

# ===== DATOS =====
with t_datos:
    resumen = pd.DataFrame([{"grupo": GRUPOS[k]["label"], "n": len(v),
                             "variables": ", ".join(v) if v else "—"} for k, v in sel.items()])
    st.dataframe(resumen, use_container_width=True, hide_index=True)
    nulos = df_model.isna().sum()
    if nulos.any():
        st.warning("Nulos: " + ", ".join(f"{k}={v}" for k, v in nulos[nulos > 0].items()))
    st.dataframe(df_model.head(20), use_container_width=True)

# ===== TENDENCIAS =====
with t_tend:
    freq = st.radio("Agregación", ["Semanal", "Mensual", "Trimestral"], index=1, horizontal=True)
    fmap = {"Semanal": "W", "Mensual": "MS", "Trimestral": "QS"}[freq]
    st.subheader("Inversión por medio a lo largo del tiempo")
    f = core.fig_inversion_apilada(df_model, media_cols, date_col, P, fmap, tipos)
    st.pyplot(f); plt.close(f)
    st.subheader("Tendencia del KPI")
    f = core.fig_kpi_trend(df_model, kpi_col, date_col, P)
    st.pyplot(f); plt.close(f)
    st.subheader("Inversión total vs KPI")
    st.caption("Dónde suben juntos y dónde se separan: las divergencias sostenidas sugieren "
               "saturación o efecto de otros factores.")
    f = core.fig_overlay(df_model, media_cols, kpi_col, date_col, P)
    st.pyplot(f); plt.close(f)

# ===== SHARES =====
with t_share:
    st.subheader("Share of Investment (SOI)")
    ts = core.tabla_share(df_model, media_cols, tipos)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        f = core.fig_share_donut(ts, P); st.pyplot(f); plt.close(f)
    with c2:
        st.dataframe(ts, use_container_width=True, hide_index=True)

    if sel["medios_online"] and sel["medios_offline"]:
        st.subheader("Mix Online vs Offline")
        sh = core.share_online_offline(df_model, media_cols, tipos)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            f = core.fig_share_online_offline(sh, P); st.pyplot(f); plt.close(f)
        with c2:
            st.metric("Online", f"{sh['Online']}%")
            st.metric("Offline", f"{sh['Offline']}%")
            st.caption("Un mix muy cargado a un lado dificulta estimar el efecto del otro.")

    st.subheader("Evolución del share por canal")
    f = core.fig_share_temporal(df_model, media_cols, date_col, P)
    st.pyplot(f); plt.close(f)

    if sel["competencia_inv"]:
        st.subheader("Share of Spend: nosotros vs competencia")
        sos = core.share_of_spend(df_model, media_cols, sel["competencia_inv"], date_col)
        if sos is not None and len(sos):
            f = core.fig_share_of_spend(sos, P); st.pyplot(f); plt.close(f)
            st.metric("Nuestro share promedio", f"{sos['Nosotros_%'].mean():.1f}%")

    if sel["competencia_share"]:
        st.subheader("Share of Voice de competencia")
        f = core.fig_sov_lineas(df_model, sel["competencia_share"], date_col, P,
                                "SOV de competencia en el tiempo")
        st.pyplot(f); plt.close(f)

# ===== ALERTAS =====
with t_alert:
    n_rojo = sum(1 for a in alertas if a["sev"] == "error")
    c1, c2 = st.columns(2)
    c1.metric("Alertas críticas 🔴", n_rojo)
    c2.metric("Para revisar 🟡", len(alertas) - n_rojo)

    st.subheader("Tablero por canal")
    def _estilo(col):
        out = []
        for v in col:
            s = ""
            if col.name == "SOI_%" and v < UMBRALES["soi_minimo_pct"]: s = "background-color:#fff3cd"
            if col.name == "%ceros" and v > UMBRALES["pct_ceros"]: s = "background-color:#fff3cd"
            if col.name == "racha_ceros" and v >= UMBRALES["racha_ceros"]: s = "background-color:#f8d7da"
            if col.name == "picos" and (v / len(df_model) * 100) > UMBRALES["pct_picos"]: s = "background-color:#ffe0b2"
            if col.name == "corr_KPI" and abs(v) < UMBRALES["corr_kpi_minima"]: s = "background-color:#fff3cd"
            out.append(s)
        return out
    try:
        st.dataframe(tablero.style.apply(_estilo), use_container_width=True)
    except Exception:
        st.dataframe(tablero, use_container_width=True)

    st.subheader("Alertas")
    if not alertas:
        st.success("✅ Sin alertas: los canales se ven sanos.")
    else:
        orden = {"error": 0, "warning": 1, "info": 2}
        for a in sorted(alertas, key=lambda x: orden.get(x["sev"], 3)):
            getattr(st, a["sev"])(f"**{a['tema']} — {a['variable']}**  \n{a['detalle']}  \n➜ {a['sugerencia']}")

    st.subheader("Distribuciones")
    cols_plot = [kpi_col] + predictores
    ncols = 3; nrows = int(np.ceil(len(cols_plot) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    axes = axes.reshape(-1)
    for ax, c in zip(axes, cols_plot):
        sns.histplot(df_model[c].dropna(), kde=True, ax=ax, color=P["secundario"])
        ax.set_title(c, fontsize=9); ax.set_xlabel("")
    for ax in axes[len(cols_plot):]:
        ax.axis("off")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("Redundancia (VIF) y correlación")
    vif = core.vif_table(df_model, predictores)
    cc1, cc2 = st.columns([1, 1.3])
    with cc1:
        st.dataframe(vif, use_container_width=True, hide_index=True)
        red = vif[vif["VIF"] > UMBRALES["vif_alerta"]]["variable"].tolist()
        if red:
            st.warning(f"VIF alto en: {red}. Se pisan entre sí; considera quitar una.")
        else:
            st.success("Sin redundancia preocupante.")
    with cc2:
        figc, axc = plt.subplots(figsize=(1.0 * len(predictores) + 2, 0.8 * len(predictores) + 2))
        sns.heatmap(df_model[predictores].corr(method="spearman"), annot=True, fmt=".2f",
                    cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=axc)
        st.pyplot(figc); plt.close(figc)

    st.subheader("¿Manda la tendencia? (ADF / KPSS)")
    try:
        import warnings; warnings.filterwarnings("ignore")
        from statsmodels.tsa.stattools import adfuller, kpss
        rows = []
        for c in [kpi_col] + predictores:
            s = df_model[c].dropna()
            try: padf = adfuller(s, autolag="AIC")[1]
            except Exception: padf = np.nan
            try: pk = kpss(s, regression="c", nlags="auto")[1]
            except Exception: pk = np.nan
            rows.append({"variable": c, "p_ADF": round(padf, 3) if pd.notna(padf) else None,
                         "p_KPSS": round(pk, 3) if pd.notna(pk) else None,
                         "tendencia_fuerte": "⚠️ sí" if (pd.notna(padf) and padf > 0.05) else "no"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("En Meridian NO se diferencia: la tendencia la maneja con knots. Esto es solo alerta.")
    except ImportError:
        st.info("Instala statsmodels para ADF/KPSS:  pip install statsmodels")

# ===== MODELADOR =====
with t_model:
    st.subheader("¿Cuánto peso esperar del total de medios?")
    pct_ts, pct_media, techo = seccionB["descomp"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tendencia + estacionalidad", f"{pct_ts}%")
    c2.metric("Medios (del residual)", f"{pct_media}%")
    c3.metric("Techo de contribución", f"~{techo}%")
    st.caption("Referencia para tus priors de ROI, no la contribución final.")

    st.subheader("Contribución esperada por canal (SOI vs eficiencia)")
    ef = core.eficiencia(tablero)
    st.dataframe(ef.round(3), use_container_width=True, hide_index=True)
    figE, axE = plt.subplots(figsize=(8, 5))
    cols_pt = {"Online": P["secundario"], "Offline": P["acento"], "—": P["primario"]}
    for _, r in ef.iterrows():
        axE.scatter(r["SOI_%"], r["fuerza_senal"], s=95,
                    color=cols_pt.get(r.get("tipo", "—"), P["primario"]))
        axE.annotate(r["canal"], (r["SOI_%"], r["fuerza_senal"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    axE.set_xlabel("SOI % (peso en inversión)"); axE.set_ylabel("Fuerza de señal (|corr| con KPI)")
    axE.set_title("Arriba-izquierda = mucha señal con poca inversión"); axE.grid(alpha=.3)
    st.pyplot(figE); plt.close(figE)

    st.subheader("Forma de curva a esperar")
    curvas, overrides = core.lectura_curvas(df_model, media_cols, kpi_col, tipos)
    st.dataframe(curvas, use_container_width=True, hide_index=True)

    st.subheader("Matriz de correlación cruzada con el KPI")
    st.caption("Correlación de cada variable con el KPI a distintos rezagos. La casilla más intensa "
               "por fila indica dónde más se relaciona: pista del adstock.")
    inc = st.multiselect("Grupos a incluir", list(GRUPOS.keys()),
                         default=["medios_online", "medios_offline"],
                         format_func=lambda k: GRUPOS[k]["label"])
    vars_ccf = [c for k in inc for c in sel[k]]
    if vars_ccf:
        m = core.ccf_matriz(df_model, vars_ccf, kpi_col, max_lag=8)
        f = core.fig_ccf_heatmap(m, P); st.pyplot(f); plt.close(f)
    else:
        st.info("Selecciona al menos un grupo con variables.")

    st.subheader("🎛️ Variables de control por grupo")
    if not ctrl_cols:
        st.info("No seleccionaste variables de control (competencia, negocio o macro).")
    else:
        cvk = core.control_vs_kpi(df_model, ctrl_cols, kpi_col, grupo_de)
        cvk["grupo"] = cvk["grupo"].map(lambda k: GRUPOS.get(k, {}).get("label", k))
        st.dataframe(cvk.round(3), use_container_width=True, hide_index=True)
        for _, r in cvk.iterrows():
            if abs(r["corr_KPI"]) >= 0.4:
                st.success(f"**{r['variable']}** ({r['grupo']}): fuerte relación con el KPI "
                           f"({r['corr_KPI']}). Buen control, mantenlo.")
            elif abs(r["corr_KPI"]) >= 0.15 or abs(r["corr_en_lag"]) >= 0.2:
                st.info(f"**{r['variable']}** ({r['grupo']}): relación moderada ({r['corr_KPI']}; "
                        f"mejor rezago {int(r['mejor_lag'])} → {r['corr_en_lag']}).")
            else:
                st.warning(f"**{r['variable']}** ({r['grupo']}): casi no se relaciona con el KPI "
                           f"({r['corr_KPI']}). Revisa si aporta como control.")
        for k in ("competencia_inv", "competencia_share", "negocio", "macro"):
            if sel[k]:
                st.markdown(f"**Tendencia · {GRUPOS[k]['label']}**")
                f = core.fig_tendencias_grupo(df_model, sel[k], date_col, P, GRUPOS[k]["label"])
                st.pyplot(f); plt.close(f)

    st.subheader("¿Tengo datos suficientes? (densidad de señal)")
    ratio, params = seccionB["densidad"]
    st.metric("Observaciones por parámetro", ratio, help=f"{len(df_model)} obs / {params} parámetros")
    if ratio >= 15: st.success("🟢 Cómodo: los datos mandan sobre los priors.")
    elif ratio >= 8: st.warning("🟡 Aceptable: los priors influyen; que sean razonables.")
    elif ratio >= 5: st.warning("🟠 Justo: tus priors pesarán bastante. Reduce canales/knots.")
    else: st.error("🔴 Insuficiente: el modelo se pegará a los priors. Agrupa canales o pasa a geo-level.")

# ===== INFORME =====
with t_inf:
    st.subheader("Informe final")
    st.write("Reúne variables, alertas, tendencias, shares y lectura para el modelador.")
    meta = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "filas": len(df_model),
            "periodo": f"{FECHA_INI} a {FECHA_FIN}"}
    figs = [("Inversión por medio", core.fig_to_b64(
                core.fig_inversion_apilada(df_model, media_cols, date_col, P, "MS", tipos))),
            ("Tendencia del KPI", core.fig_to_b64(core.fig_kpi_trend(df_model, kpi_col, date_col, P))),
            ("Inversión total vs KPI", core.fig_to_b64(
                core.fig_overlay(df_model, media_cols, kpi_col, date_col, P))),
            ("Share of Investment", core.fig_to_b64(
                core.fig_share_donut(core.tabla_share(df_model, media_cols, tipos), P))),
            ("Evolución del share", core.fig_to_b64(
                core.fig_share_temporal(df_model, media_cols, date_col, P)))]
    if sel["medios_online"] and sel["medios_offline"]:
        figs.append(("Mix Online vs Offline", core.fig_to_b64(core.fig_share_online_offline(
            core.share_online_offline(df_model, media_cols, tipos), P))))
    if sel["competencia_inv"]:
        sos = core.share_of_spend(df_model, media_cols, sel["competencia_inv"], date_col)
        if sos is not None and len(sos):
            figs.append(("Share of Spend vs competencia", core.fig_to_b64(core.fig_share_of_spend(sos, P))))

    grupos_resumen = {GRUPOS[k]["label"]: ", ".join(v) for k, v in sel.items()}
    html = core.generar_informe_html(CLIENTE, meta, alertas, tablero, figs, seccionB, P, grupos_resumen)
    st.download_button("⬇️ Descargar informe (HTML)", html,
                       file_name=f"informe_eda_{datetime.now():%Y%m%d}.html", mime="text/html")
    st.caption("Ábrelo en el navegador y usa Imprimir → Guardar como PDF si lo necesitas en PDF.")
    components.html(html, height=650, scrolling=True)

# ===== CÓDIGO =====
with t_code:
    st.subheader("Código listo para pegar en Colab")
    st.write("`CONFIG` va en la Parte 1.2-bis, `HP_OVERRIDES` en `HP['overrides']` (Parte 4) "
             "y `START`/`END` en la Parte 1.3.")
    _, overrides = core.lectura_curvas(df_model, media_cols, kpi_col, tipos)
    codigo = core.generar_codigo_colab(CONFIG, UMBRALES, overrides, FECHA_INI, FECHA_FIN)
    st.code(codigo, language="python")
    st.download_button("⬇️ Descargar config_colab.py", codigo,
                       file_name="config_colab.py", mime="text/x-python")

st.markdown(f'<div class="csa-foot">Consumer Science &amp; Analytics · Herramienta interna para {CLIENTE}</div>',
            unsafe_allow_html=True)
