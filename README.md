# Recreación de un artículo de ML sobre valoración inmobiliaria con los datos publicados por los autores

Proyecto final del curso de Machine Learning II (Maestría en Analítica Aplicada / Inteligencia Artificial, Universidad de La Sabana).

**Artículo recreado:** Tapia, Chavez-Garzon, Pezoa, Suarez-Aldunate y Pilleux (2025), *Comparing automated valuation models for
real estate assessment in the Santiago Metropolitan Region: A study on machine learning algorithms and hedonic pricing with
spatial adjustments*, PLOS ONE 20(3): e0318701 · https://doi.org/10.1371/journal.pone.0318701

**Relación con el trabajo de grado:** modelo de IA para el avalúo predictivo de apartamentos en Bogotá y Medellín. El artículo es un
antecedente directo (modelo de valoración automática con LightGBM + SHAP frente a un modelo hedónico con ajuste espacial).

**Datos:** los autores publicaron los suyos (Mendeley Data, https://doi.org/10.17632/jdf7rzmnvw.1), 52.039 anuncios. El enunciado permite simular
los datos sólo cuando el artículo no los publica, así que aquí **se usan los datos reales y no se simula nada**.

## Estructura del repositorio
```
proyecto_ml_final/
├── README.md
├── requirements.txt
├── data/real/data_paper_plosone_set.csv   # datos reales de los autores, con la columna `set` Training/Test
├── notebooks/
│   ├── 01_articulo_y_datos.ipynb          # Etapas 1-3: artículo, comprensión y características de los datos
│   ├── 02_analisis_articulo.ipynb         # Etapas 5-6: recreación del análisis y comparación con el artículo
│   ├── 03_extension_experimental.ipynb    # Etapa 7: extensión (tamaño de muestra)
│   └── colab/                             # COMPLEMENTO: código de los autores adaptado (Colab o ejecución local)
│       ├── lgbm_section_colab.ipynb       #   modo rápido (GBDT) y completo (DART 8.000 iteraciones, horas)
│       └── ols_sar_section_colab.ipynb
├── src/
│   ├── config.py         # cifras reportadas por el artículo (Tablas 3, 5-10) y listas de variables
│   ├── datos_reales.py   # carga de los datos reales
│   ├── espacial.py       # matriz W KNN, rezago espacial, I de Moran
│   ├── modelos.py        # métricas, LightGBM, OLS, SAR (2SLS), validación cruzada
│   ├── analisis.py       # funciones de alto nivel usadas por los notebooks
│   └── figuras.py        # gráficos
├── results/              # tablas en CSV (comparación con el artículo, secuencia, ranking, extensión)
├── figures/              # gráficos en PNG
└── references/           # artículo (PDF), enunciado y notas
```
La Etapa 4 (simulación) no aplica porque se tienen los datos reales (notebook 01); las conclusiones (Etapa 8) están al final del notebook 03 y en la presentación.

## Cómo ejecutarlo
```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_articulo_y_datos.ipynb       # ~1 min
jupyter nbconvert --to notebook --execute --inplace 02_analisis_articulo.ipynb       # ~15-20 min con 2 núcleos
jupyter nbconvert --to notebook --execute --inplace 03_extension_experimental.ipynb  # ~10-15 min con 2 núcleos
```
Los tiempos son de una máquina de 2 núcleos; un portátil moderno tarda menos. Los notebooks vienen ya ejecutados; volver a correrlos es opcional.
Todo es reproducible: la validación cruzada usa la semilla 123 del código de los autores y la partición Training/Test es la columna `set` del archivo.

### En Windows con VS Code
1. Abra la carpeta `proyecto_ml_final` en VS Code e instale las extensiones **Python** y **Jupyter**.
2. En el terminal integrado: `python -m venv venv`, luego `.\venv\Scripts\Activate.ps1` y `pip install -r requirements.txt`
   (si PowerShell bloquea la activación: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`).
3. Abra cada notebook, elija como kernel el entorno `venv` y ejecute las celdas en el orden 01 → 02 → 03.
   Los notebooks se ejecutan desde la carpeta `notebooks/` (VS Code lo hace por defecto). Abra VS Code en la carpeta raíz del proyecto.

### Notebooks de `notebooks/colab/` (complemento)
Son las secciones LightGBM y OLS/SAR del código de los autores, adaptadas (sin el error de `verbose` de LightGBM 4, con SHAP sobre una
muestra y matriz W dispersa). Tienen modo `rapido` (GBDT, minutos) y `completo` (DART con 8.000 iteraciones, horas), que es el único que replica exactamente el
modelo del artículo. Sus resultados **no** están en `results/`.

## Resultados principales (detalle en `results/` y en los notebooks)
LightGBM: hiperparámetros de los autores pero GBDT con 400 árboles (no DART con 8.000), lo que explica gran parte de la diferencia con el artículo.

| Resultado | Artículo | Datos reales (este proyecto) |
|---|---|---|
| LightGBM, Training/Test: MAPE / MdAPE | 10,04 % / 6,77 % | 10,81 % / 7,57 % |
| LightGBM, Training/Test: MAE / RMSE (UF) | 798 / 1.359 | 855 / 1.405 |
| LightGBM, Training/Test: R² (UF) | 0,9525 | 0,9492 |
| Secuencia de modelos, MAPE 5-fold (base → final) | 11,59 % → 10,07 % | 11,74 % → 10,85 % |
| Efecto de las 10 variables de fotos sobre el MAE | −1,2 % | −0,03 % |
| OLS, 5-fold: MAPE / R² (UF) | 21,13 % / 0,710 | 21,85 % / 0,683 |
| SAR, 5-fold (forma reducida): MAPE / R² (UF) | 20,92 % / 0,741 | 22,20 % / 0,637 |
| SAR, 5-fold (con precios de vecinos): MAPE / R² (UF) | — | 20,50 % / 0,720 |
| I de Moran: ln(precio) / residuos OLS / residuos SAR | 0,711 / 0,134 / 0,075 | 0,711 / 0,142 / 0,079 |
| R² en muestra (log): OLS / SAR | 0,865 / 0,877 | 0,866 / 0,879 |
| rho (SAR) | 0,427 | 0,435 |
| Signos de los coeficientes SAR iguales a la Tabla 8 | — | 33 de 33 (con la constante) |
| Top-5 de importancia que coincide con el artículo (LightGBM / OLS / SAR) | — | 5 / 5 / 5 |

**Extensión (una sola condición: tamaño de muestra, n = 1.000 a 52.039, 3 repeticiones por tamaño):** el MAPE de LightGBM sube de 10,85 % a 16,67 % al reducir
la muestra; el del mejor econométrico pasa de 21,85 % a 22,80 %. LightGBM sigue siendo mejor en toda la rejilla, pero su ventaja baja de 50,3 % a 26,9 %.

## Supuestos, desviaciones y limitaciones (leer antes de usar los resultados)
- **Datos:** son los publicados por los autores en Mendeley Data. **No se verificó la licencia** del conjunto: si este repositorio va a ser público, confírmela
  o retire `data/real/` y deje sólo el enlace al DOI (el código funciona igual si usted descarga el archivo y lo coloca en esa ruta).
  El conjunto tiene 49 comunas con anuncios (el artículo menciona 52) y el máximo de superficie de casas es 2.400 m² (el artículo reporta un filtro hasta 2.500).
- **Qué se reproduce:** I de Moran de ln(precio) (0,7108 vs 0,7107), R² y rho de la Tabla 8, los 33 signos de los coeficientes SAR y el hallazgo de los
  colegios cercanos (OLS −0,0008, SAR +0,0014 en el artículo; −0,0008 y +0,0015 aquí).
- **LightGBM:** se usaron los hiperparámetros del código de los autores (`lgbm_section.ipynb`: tasa 0,1, 256 hojas, `max_depth` 40, `max_bin` 512, `min_data_in_leaf` 10,
  `feature_fraction` 0,9, `bagging_fraction` 0,7) pero con GBDT y 400 árboles en lugar de DART con 8.000 iteraciones y `drop_rate` 0,12, para que los notebooks corran en minutos.
  Por eso el MAPE es ≈ 0,8 puntos mayor que el del artículo y las fotos casi no mejoran el error (−0,03 % del MAE frente a −1,2 %). El modo `completo` de `notebooks/colab/` ejecuta DART.
- **SAR:** estimado por mínimos cuadrados en dos etapas espaciales, igual que el código de los autores (`spreg.GM_Lag`); aquí los instrumentos son X, WX y W²X (los autores sólo WX).
  OLS usa errores robustos HC1 (los autores, `robust='white'` de spreg). El objetivo es log1p(precio), como en el código de los autores.
- **Predicción fuera de muestra del SAR:** el artículo no explica cómo se hace. Con la forma reducida (sólo X y coordenadas), el SAR queda peor que OLS (MAPE 22,2 % vs 21,8 %), contrario al artículo.
  Con el promedio del ln(precio) de los vecinos de entrenamiento (variante de la sección 5d del notebook 02, elección de este proyecto) queda mejor que OLS (20,5 %, R² 0,72), más cerca del artículo (20,9 %, 0,74).
  Las tablas principales y la extensión usan la forma reducida.
- **Selección de variables:** no se repite la selección variable por variable; se usa el conjunto final de la Tabla 4 y la secuencia de la Tabla 3.
- **Validación aleatoria:** con autocorrelación espacial fuerte puede ser optimista; la validación espacial (OE3 del trabajo de grado) queda como trabajo pendiente.
- Las métricas de razón (mediana de la razón, COD, PRD) son un **complemento** para el OE4; no están en el artículo. Los rangos de referencia de la literatura sobre estudios de razón no se verificaron en este proyecto.
