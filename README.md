# App de Preproceso + EDA para MMM (Google Meridian)
### Consumer Science & Analytics · Cliente: Tigo–Millicom

App de Streamlit para cargar la base, **clasificar las variables**, revisar **tendencias,
shares y alertas**, y generar el **código listo para Meridian en Colab**.
No entrena el modelo (eso se queda en Colab/GPU): es liviana y rápida.

---

## Estructura del repo

```
.
├── app.py                 # interfaz (Streamlit)
├── core.py                # lógica de datos, estadísticos y gráficas
├── requirements.txt       # dependencias
├── logo.png               # logo del cliente  ← ponlo tú
├── .gitignore
└── .streamlit/
    └── config.toml        # tema base de Streamlit
```

> **Logo:** la app lo busca en `logo.png` (raíz). Si prefieres `assets/logo.png`,
> cambia la constante `LOGO_PATH` al inicio de `app.py`.

> **Datos:** el `.gitignore` bloquea `.xlsx`/`.csv` a propósito — las bases del
> cliente no deben subirse al repositorio.

---

## Cómo correrla en local

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en http://localhost:8501

## Cómo publicarla (Streamlit Community Cloud, gratis)

1. Sube este repo a GitHub.
2. Entra a [streamlit.io/cloud](https://streamlit.io/cloud) e inicia sesión con GitHub.
3. **Create app** → elige el repo, la rama (`main`) y el archivo `app.py`.
4. **Deploy**. Cada `git push` reconstruye la app automáticamente.

---

## Qué hace

### 🎨 Colores configurables
Selector de paletas predefinidas (Tigo–Millicom, CSA, Océano, Bosque, Violeta, Grafito)
o ajuste manual de los colores **primario / secundario / acento**. El tema se aplica al
banner, a todas las gráficas y al informe descargable.

### ⚙️ Clasificación de variables
Los campos se configuran en el panel principal (no en el sidebar), en seis grupos:

| Grupo | Entra a Meridian como |
|---|---|
| Medios **ONLINE** (inversión) | media |
| Medios **OFFLINE** (inversión) | media |
| Inversión de **COMPETENCIA** | control |
| **SHARE** de competencia (SOV) | control |
| Variables de **NEGOCIO** | control |
| Variables **MACROECONÓMICAS** | control |

Con bases de muchas columnas, activa el filtrado por patrón de texto (`INVR`, `POBLAC`, …).

### 🗓️ Recortador de fechas
Slider de rango que acota el periodo; **todo** (alertas, gráficas, informe y el código
exportado) se recalcula sobre el tramo seleccionado.

### Pestañas
- **📄 Datos** — resumen por grupo, nulos y vista previa.
- **📈 Tendencias** — inversión apilada por medio (semanal/mensual/trimestral), tendencia
  del KPI e inversión total vs KPI.
- **🥧 Shares** — SOI por canal (donut), mix Online vs Offline, evolución del share,
  Share of Spend frente a la competencia y SOV de competencia.
- **🚨 Alertas** — tablero con semáforo y alertas en lenguaje claro (SOI bajo, silencios,
  apagones, picos atípicos, relación débil, mix desbalanceado), distribuciones, VIF y ADF/KPSS.
- **🎯 Modelador** — techo de contribución, SOI vs eficiencia, forma de curva por canal,
  matriz de correlación cruzada con el KPI por rezago, y análisis de los controles por grupo.
- **📑 Informe** — documento HTML descargable con variables, alertas, gráficas y lectura
  para el modelador (imprímelo a PDF si lo necesitas).
- **🧩 Código Colab** — bloque `CONFIG` + `UMBRALES` + `HP_OVERRIDES` + `START`/`END`.

---

## Puente con el notebook de Colab

En la pestaña **Código Colab** copias el bloque generado y lo pegas en el notebook:

| Bloque | Dónde va en el notebook |
|---|---|
| `CONFIG` | Parte 1.2-bis |
| `START` / `END` | Parte 1.3 |
| `HP_OVERRIDES` | `HP['overrides']` en la Parte 4 |

El entrenamiento de Meridian (MCMC, pesado, ideal con GPU) se hace allá.
