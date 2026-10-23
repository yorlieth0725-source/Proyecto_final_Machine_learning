# Notas del artículo de referencia

**Tapia J, Chavez-Garzon N, Pezoa R, Suarez-Aldunate P, Pilleux M (2025).** Comparing automated valuation models for real
estate assessment in the Santiago Metropolitan Region: A study on machine learning algorithms and hedonic pricing with
spatial adjustments. *PLOS ONE* 20(3): e0318701. https://doi.org/10.1371/journal.pone.0318701 (acceso abierto, CC BY 4.0).

**Datos y código de los autores:** Mendeley Data, "Replication data and Python programs...", versión 1, 25 de julio de 2024.
DOI 10.17632/jdf7rzmnvw.1 · https://data.mendeley.com/datasets/jdf7rzmnvw/1. De ahí se descargaron `data_paper_plosone.csv` y
`data_paper_plosone_set.csv` (52.039 anuncios; el segundo agrega la columna `set` Training/Test). Este repositorio incluye una copia de
`data_paper_plosone_set.csv` en `data/real/`; la licencia del conjunto no se verificó.

**Material complementario en PLOS ONE:** S1, S2 y S3 Table (PDF), DOI 10.1371/journal.pone.0318701.s001 a .s003.

## Fuentes externas que usan los autores
- ICVU (Índice de Calidad de Vida Urbana): Orellana et al. (2012), Pontificia Universidad Católica de Chile.
- SIMCE (resultados de calidad educativa de colegios, Chile).
- UF (Unidad de Fomento): Banco Central de Chile.
- Imágenes para el clasificador (DenseNet201, 16 clases): SUN397, Places, CubiCasa5k e imágenes de Coderhub SpA.

## Dónde está cada cifra reportada que se usa en `src/config.py`
| Contenido | Ubicación en el artículo |
|---|---|
| N = 52.039; filtros; variables | Sección 1.2.2, Tabla 1 |
| Variables del modelo final (34) | Tabla 4 |
| Métricas | Tabla 2 |
| Secuencia de modelos (MAPE, MdAPE, MAE, RMSE, R²) | Tabla 3 |
| Modelo final, partición 80/20 | Tabla 5 |
| I de Moran de ln(precio) | Tabla 6 |
| I de Moran de residuos OLS y SAR | Tabla 7 |
| Coeficientes OLS y SAR, R², rho | Tabla 8 |
| OLS y SAR en 5-fold CV | Tabla 9 |
| Ranking de importancia | Tabla 10 |
