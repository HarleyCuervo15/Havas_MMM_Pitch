# Modo Pitch — MMM econométrico rápido

App complementaria a la de EDA. Estima un modelo direccional en segundos:
**adstock geométrico + saturación cóncava + Ridge con no-negatividad**, con
búsqueda aleatoria de hiperparámetros y validación de origen móvil.

```bash
pip install streamlit pandas numpy scipy scikit-learn matplotlib seaborn
streamlit run app_pitch_mmm.py
```

---

## Cómo se optimiza el R²

La búsqueda muestrea combinaciones de `theta` (adstock), `alpha` (saturación) y
`lambda` (regularización), ajusta el modelo y se queda con la mejor según el
criterio que elijas:

| Criterio | Cuándo usarlo |
|---|---|
| **Mixto** (por defecto) | Promedia R² ajustado y R² fuera de muestra. Es el que evita sobreajuste. |
| R² ajustado | Cuando solo quieres el mejor ajuste histórico y tienes pocas semanas. |
| R² fuera de muestra | Cuando el pitch va a hablar de capacidad predictiva. |

Después del muestreo aleatorio corre un **refinamiento local**: mueve cada
`theta` y `alpha` en pasos de 0.10 y 0.05 y reescala `lambda`, quedándose solo
con las mejoras. Ahí es donde se exprimen los últimos puntos de R².

Tres cosas deliberadas sobre el R²:

1. **El R² fuera de muestra se mide contra el predictor ingenuo** (la media del
   tramo de entrenamiento), no contra la media de la ventana. Con un KPI plano
   la media de una ventana de 12 semanas es casi el valor mismo y el R²
   tradicional se desploma sin que el modelo empeore.
2. **La banda de contribución creíble** penaliza levemente las combinaciones que
   ganan R² a costa de una descomposición absurda (medios explicando 60% del
   KPI). Es un filtro, no una imposición.
3. **La pestaña Candidatos** muestra los 15 mejores modelos. Si varios tienen R²
   casi igual pero descomposiciones muy distintas, esa dispersión es la
   incertidumbre real que el R² no te está contando: vale más que el decimal.

**Los rangos por canal son el sustituto de los priors.** Es la palanca que más
mueve la calidad: acotar `theta` de search a 0.0–0.25 y el de TV a 0.5–0.85
recupera bastante mejor el reparto entre canales que dejarlos libres.

---

## Ejemplo 1 · `ejemplo_1_base_limpia.csv`

156 semanas, KPI `VENTAS_ACTIVACIONES` con tendencia y estacionalidad claras,
5 canales, 3 controles. Caso de manual.

**Verdad con la que se generó:**

| canal | theta | alpha | contribución real |
|---|---|---|---|
| INVR_TV | 0.70 | 0.40 | 11.4% |
| INVR_DIGITAL_VIDEO | 0.50 | 0.60 | 5.7% |
| INVR_SEARCH | 0.15 | 0.85 | 5.2% |
| INVR_SOCIAL | 0.35 | 0.70 | 3.1% |
| INVR_RADIO | 0.60 | 0.50 | 2.1% |
| **total medios** | | | **27.5%** |

Controles reales: precio con efecto negativo, cobertura positiva, festivos positivos.

**Qué deberías obtener** (medios por patrón `INVR`, controles `PRECIO_INDICE`,
`POBLAC_COBERTURA`, `FESTIVOS`, 2 armónicos, 400 iteraciones):
R² ≈ 0.89 · R² ajustado ≈ 0.88 · R² fuera de muestra ≈ 0.88 · MAPE ≈ 2.5% ·
contribución de medios ≈ 25%.

Con los rangos por canal puestos a mano el reparto se acerca bastante más a la
verdad (search queda casi exacto, TV sube de ~5% a ~8%).

---

## Ejemplo 2 · `ejemplo_2_base_dificil.csv`

130 semanas, KPI `VENTAS_NETAS` de un mercado maduro. Está construido con todas
las trampas que pediste:

- **KPI casi constante**: coeficiente de variación 0.048 y tendencia casi nula.
- **Dos controles con efecto negativo**: `ARPU_PRECIO` y `PRESION_COMPETENCIA`.
- **Inversión reactiva**: `INVR_RETENCION` sube cuando las ventas caen la semana
  anterior. Su correlación cruda con el KPI es **−0.28** aunque su efecto real es
  positivo y es el canal más eficiente de todos.
- **Colinealidad**: `INVR_SEARCH` y `INVR_SOCIAL` correlacionan 0.77 entre sí.
- **Canal intermitente**: `INVR_OOH` tiene 69 semanas en cero y efecto casi nulo.
- **Una semana atípica** (semana 72) con promoción agresiva.

**Verdad con la que se generó:**

| canal | theta | alpha | contribución real | corr. cruda con el KPI |
|---|---|---|---|---|
| INVR_TV | 0.75 | 0.35 | 4.7% | +0.14 |
| INVR_SEARCH | 0.10 | 0.90 | 1.9% | +0.29 |
| INVR_SOCIAL | 0.30 | 0.75 | 1.7% | +0.37 |
| INVR_OOH | 0.55 | 0.55 | 0.3% | +0.19 |
| **INVR_RETENCION** | 0.25 | 0.65 | **2.8%** | **−0.28** |
| **total medios** | | | **11.3%** | |

**Qué deberías obtener:** R² ≈ 0.27 · R² ajustado ≈ 0.18 · R² fuera de muestra
≈ 0.12 · MAPE ≈ 2.6% · contribución de medios ≈ 9.5%.

Ese R² bajo **no es un fallo del modelo**: con un KPI plano no hay varianza que
explicar, y el MAPE de 2.6% dice que la predicción es buena en términos
relativos. Es justo el caso donde presentar solo el R² en un pitch se vuelve
engañoso en las dos direcciones.

Lo que hay que observar en este ejemplo:

- **`INVR_RETENCION` se va a cero o queda subestimado.** La restricción de
  no-negatividad lo frena en 0 en vez de dejarlo salir negativo. El aviso
  "Canales en cero" te lo señala. Sin priors, un modelo frecuentista no puede
  desenredar una inversión que responde al KPI en vez de causarlo: hay que
  modelarlo aparte o pasar a Meridian.
- **Search y Social se roban crédito entre sí** de una corrida a otra. Cambia la
  semilla y compara: si el reparto entre ellos baila, no lo presentes separado,
  preséntalos agrupados como "digital performance".
- **TV tiende a quedar sobreestimado** porque su serie suavizada se parece a la
  base. Acotar su `alpha` a 0.25–0.55 ayuda.

---

## Límites que conviene decir en el pitch

Es un modelo frecuentista regularizado, sin priors. Sirve para dimensionar el
tamaño de la oportunidad, ordenar canales por eficiencia y mostrar dónde hay
saturación. No sirve para comprometer una asignación de presupuesto: con medios
colineales el reparto entre canales se mueve. Esa brecha es exactamente el
argumento para pasar a Meridian en la fase de delivery.

---
---

# v2 — Recomendador, benchmark, dummies y alerta de R²

Archivo: `app_pitch_mmm_v2.py` (la v1 sigue funcionando; la v2 la reemplaza).

## Flujo recomendado

1. **🧭 Analizar y recomendar** — barre theta y alpha canal por canal y carga
   rangos y puntos iniciales en los controles. También decide cuántos armónicos
   de estacionalidad y si conviene tendencia cuadrática.
2. **⚡ Estimar modelo** — primera corrida.
3. **🏷️ Dummies** — revisa las sugeridas, aplica las marcadas como *explica* y
   vuelve a estimar. Aquí es donde sube el R².
4. **🎯 Benchmark** — ancla la contribución al MMM previo o al benchmark de
   categoría. Aquí es donde se ordena la descomposición.

## Qué aporta cada palanca (medido sobre las bases de ejemplo)

`error` = desvío promedio en puntos porcentuales entre la contribución estimada
por canal y la real con la que se generó la base.

**Ejemplo 1 (limpia)**

| paso | R² | R² fuera | medios % (real 27.5) | error |
|---|---|---|---|---|
| sin nada | 0.892 | 0.892 | 25.2% | 3.26 pp |
| + recomendador | 0.891 | 0.892 | 33.8% | 3.69 pp |
| + 2 dummies | 0.905 | 0.899 | 39.8% | 5.84 pp |
| + benchmark | 0.901 | 0.891 | **27.7%** | **0.65 pp** |

**Ejemplo 2 (difícil)**

| paso | R² | R² fuera | medios % (real 11.3) | error |
|---|---|---|---|---|
| sin nada | 0.287 | 0.114 | 13.3% | 0.80 pp |
| + recomendador | 0.301 | 0.125 | 17.3% | 2.21 pp |
| + 4 dummies | **0.678** | 0.217 | 14.6% | 2.26 pp |
| + benchmark | 0.655 | 0.202 | **10.2%** | 0.99 pp |

Tres lecturas que conviene tener presentes:

- **Las dummies son la palanca del R²**, no el recomendador. En la base difícil
  el R² pasa de 0.30 a 0.68 y cruza el umbral del 50%.
- **El benchmark es la palanca de la descomposición.** En la base limpia baja el
  error de 5.84 a 0.65 pp cediendo apenas 0.004 de R².
- **Las dummies inflan la contribución de medios si nadie la ancla** (25% → 40%
  en la base limpia). Por eso la tabla de dummies muestra `Δmedios_pp` y por eso
  conviene correr dummies y benchmark juntos.

## Puntos iniciales y anclaje

Cada canal tiene, además del rango, un **θ0 y α0**. Ese punto se evalúa siempre
con tres niveles de regularización y es desde donde arranca el refinamiento
local, así que una corrida nunca sale peor que tu hipótesis de partida. El
deslizador de **anclaje** restringe además el muestreo aleatorio a un radio
alrededor de ese punto: úsalo cuando ya tengas un modelo previo y solo quieras
reajustar.

## Benchmark de contribución

Penalización blanda sobre la desviación respecto a los objetivos que declares
(total y/o por canal). El peso va de 0 a 3: con 1.5 la solución se pega bastante
sin destruir el ajuste. Si al subir el peso el R² se cae en picada, los datos
están discrepando del benchmark y eso mismo es un hallazgo para el pitch.

## Dummies

- **Manuales**: fecha puntual, periodo, escalón desde una fecha, o meses.
- **Sugeridas por residuales**: picos atípicos (z robusto), tramos con sesgo
  sostenido, un cambio de nivel y meses mal explicados.

Cada candidata se prueba en el modelo actual y se reporta con tres columnas:

| columna | qué dice |
|---|---|
| `ΔR2_adj` | cuánto sube el ajuste histórico |
| `ΔR2_fuera` | si además mejora fuera de muestra |
| `Δmedios_pp` | cuánto desplaza la contribución total de medios |

El veredicto combina las tres: *explica* (se selecciona por defecto),
*mueve la descomposición* (sube el R² pero reasigna más de 4 pp de crédito) y
*solo tapa el dato* (sube el R² dentro de muestra y empeora fuera).

En la base difícil la primera sugerencia es `D_pico_2025-05-12`, que es
exactamente la semana de promoción atípica con la que se generó el archivo.

## Alerta de R²

El umbral es configurable (50% por defecto). Si el R² no llega, la alerta es
crítica y dice cuántos puntos faltan, recuerda mirar el MAPE cuando el KPI es
plano y avisa si aún no has aplicado dummies. Entre el umbral y +10 puntos sale
una advertencia amarilla. También hay alerta cuando el ajuste no se traslada
fuera de muestra teniendo dummies aplicadas.

## Detección de inversión reactiva

El recomendador compara la inversión de cada semana contra el KPI de la semana
anterior. Si la correlación es muy negativa, marca el canal: el presupuesto está
respondiendo al desempeño en vez de causarlo. En la base difícil detecta
`INVR_RETENCION` (−0.57) y ninguno en la base limpia. Ese canal no se puede
estimar sin priors: o lo anclas con benchmark o lo sacas del bloque de medios.
