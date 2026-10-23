"""Parámetros y resultados reportados en el artículo de referencia.

Artículo: Tapia J, Chavez-Garzon N, Pezoa R, Suarez-Aldunate P, Pilleux M (2025).
"Comparing automated valuation models for real estate assessment in the Santiago
Metropolitan Region: A study on machine learning algorithms and hedonic pricing with
spatial adjustments". PLOS ONE 20(3): e0318701. https://doi.org/10.1371/journal.pone.0318701
Datos y código de los autores: https://doi.org/10.17632/jdf7rzmnvw.1 (Mendeley Data)

Todo lo que está en REPORTADO_* se copió de las tablas del artículo (se indica cuál).
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "data"
DIR_RESULTADOS = RAIZ / "results"
DIR_FIGURAS = RAIZ / "figures"

SEMILLA = 42
SEMILLA_CV = 123        # semilla de la validación cruzada del código de los autores

# ---------------------------------------------------------------- Características reportadas
N_FINAL = 52_039                 # propiedades usadas (sección 1.2.2)
N_INICIAL = 98_630               # propiedades antes de depurar
PRECIO_MIN_UF, PRECIO_MAX_UF = 1_000, 27_400   # filtro de precio (sección 1.2.2)
LN_PRECIO_MEDIO = 8.8            # valor base SHAP, E[f(X)] = 6.633 UF (Fig 3)
K_VECINOS = 700                  # k del KNN para la matriz W (sección 2.2)
RHO_REPORTADO = 0.4271           # Tabla 8

# Variables del modelo final (Tabla 4): 34 predictoras
VARS_BASE = ["bathrooms", "bedrooms", "latitude", "longitude", "building_type",
             "seller_type", "total_surface"]
VARS_AMENIDADES = ["parking_lots", "swimpool", "terrace", "storage_units"]
VARS_DIST = ["dist_subway", "dist_park", "dist_hospital", "dist_police", "dist_university",
             "dist_mall", "dist_penitentiary", "dist_industry", "dist_dump"]
VARS_FOTO = ["photo_bathroom", "photo_bedroom", "photo_livingroom", "photo_diningroom",
             "photo_interior", "photo_exterior", "photo_balcony", "photo_laundryroom",
             "photo_floorplans", "photo_swimpool"]
VARS_FOTO_NO_USADAS = ["photo_kitchen", "photo_patio", "photo_gym", "photo_gameroom",
                       "photo_street", "photo_parkinglot"]
VARS_FINAL = (VARS_BASE + ["quality_municipality"] + VARS_AMENIDADES + ["new_house", "simce_4b",
              "nearby_schools_3km"] + VARS_DIST + VARS_FOTO)
assert len(VARS_FINAL) == 34

# Secuencia de la Tabla 3: (nombre, variables que se agregan)
SECUENCIA_TABLA3 = [
    ("Modelo base", VARS_BASE),
    ("M1 +quality_municipality", ["quality_municipality"]),
    ("M2 +amenities", VARS_AMENIDADES),
    ("M3 +new_house", ["new_house"]),
    ("M4 +simce_4b", ["simce_4b"]),
    ("M5 +nearby_schools_3km", ["nearby_schools_3km"]),
    ("M6 +9 distancias", VARS_DIST),
    ("Final +10 fotos", VARS_FOTO),
]

# ---------------------------------------------------------------- Resultados reportados
# Tabla 3 (validación cruzada 5-fold, LightGBM). Métricas en UF.
COLS_METRICAS = ["MAPE", "MdAPE", "MAE", "RMSE", "R2"]
REPORTADO_TABLA3 = {
    "Modelo base":              (11.59, 7.87, 906.66, 1513.44, 0.9412),
    "M1 +quality_municipality": (11.46, 7.81, 898.40, 1501.93, 0.9421),
    "M2 +amenities":            (10.94, 7.46, 863.40, 1448.80, 0.9461),
    "M3 +new_house":            (10.88, 7.44, 860.92, 1446.02, 0.9463),
    "M4 +simce_4b":             (10.72, 7.30, 849.60, 1430.32, 0.9475),
    "M5 +nearby_schools_3km":   (10.63, 7.23, 844.36, 1424.25, 0.9479),
    "M6 +9 distancias":         (10.24, 6.88, 816.99, 1391.43, 0.9503),
    "Final +10 fotos":          (10.07, 6.84, 807.53, 1373.23, 0.9516),
}
# Tabla 5: modelo final, partición simple 80/20
REPORTADO_TABLA5 = (10.04, 6.77, 797.99, 1358.80, 0.9525)
# Tabla 9: OLS y SAR en 5-fold CV
REPORTADO_TABLA9 = {"OLS": (21.13, 16.86, 1887.85, 3356.50, 0.7103),
                    "SAR": (20.92, 17.13, 1893.09, 3115.85, 0.7405)}
# Tabla 6 y 7: I de Moran
REPORTADO_MORAN = {"ln(price)": 0.7107, "residuos OLS": 0.1337, "residuos SAR": 0.0751}
# Tabla 8: R2 dentro de muestra (log-precio) y coeficientes no estandarizados
REPORTADO_R2_INSAMPLE = {"OLS": 0.8654, "SAR": 0.8774}
REPORTADO_COEF = {  # variable: (OLS, SAR)
    "constant": (3.7782, 1.9507), "bathrooms": (0.1798, 0.1712), "bedrooms": (0.0814, 0.0752),
    "building_type": (0.0874, 0.0957), "seller_type": (-0.0662, -0.0649),
    "total_surface": (0.0008, 0.0007), "quality_municipality": (0.0273, 0.0178),
    "parking_lots": (0.0159, 0.0139), "swimpool": (0.0002, 0.0021), "terrace": (0.0582, 0.0465),
    "storage_units": (0.0304, 0.0289), "new_house": (0.0753, 0.0668), "simce_4b": (0.0082, 0.0036),
    "nearby_schools_3km": (-0.0008, 0.0014), "dist_subway": (0.0045, 0.0054),
    "dist_park": (-0.0019, 0.0087), "dist_hospital": (0.0038, -0.0084),
    "dist_police": (0.0263, 0.0194), "dist_university": (-0.0176, -0.0121),
    "dist_mall": (-0.0039, 0.0052), "dist_penitentiary": (-0.0050, -0.0035),
    "dist_industry": (0.0096, -0.0006), "dist_dump": (0.0019, -0.0045),
    "photo_bathroom": (-0.0506, -0.0450), "photo_bedroom": (-0.0015, -0.0015),
    "photo_livingroom": (0.0492, 0.0438), "photo_diningroom": (0.0488, 0.0441),
    "photo_interior": (0.0017, 0.0036), "photo_exterior": (-0.0011, -0.0004),
    "photo_balcony": (0.0249, 0.0209), "photo_laundryroom": (-0.0326, -0.0249),
    "photo_floorplans": (0.0445, 0.0422), "photo_swimpool": (-0.0092, -0.0115),
}
# Tabla 10: ranking de importancia (top 5), sin latitud/longitud
REPORTADO_TOP5 = {
    "LightGBM+SHAP": ["total_surface", "quality_municipality", "simce_4b", "bathrooms", "dist_university"],
    "OLS": ["quality_municipality", "bathrooms", "total_surface", "simce_4b", "bedrooms"],
    "SAR": ["bathrooms", "quality_municipality", "total_surface", "bedrooms", "simce_4b"],
}
