"""Funciones de alto nivel que usan los notebooks 02 y 03 (recreación y extensión)."""
import numpy as np
import pandas as pd
import shap

from . import config as C
from .espacial import construir_W_knn, lat_lon_a_km, moran_I
from .modelos import (PARAMS_LGBM, VARS_ECON, cv_lightgbm, cv_ols_sar, holdout_lightgbm,
                      mejora_en_n_metricas, metricas, metricas_razon, ols_fit, sar_2sls)


# ------------------------------------------------------------------ Tabla 3: secuencia de modelos
def tabla_secuencia(df, params=PARAMS_LGBM, k=5):
    """Recrea la Tabla 3: modelos anidados evaluados con 5-fold CV. Devuelve lo obtenido (columnas `MAPE`...) y lo reportado (`MAPE_art`...)."""
    filas, vars_act, previa = [], [], None
    for nombre, nuevas in C.SECUENCIA_TABLA3:
        vars_act = vars_act + nuevas
        met, _ = cv_lightgbm(df, vars_act, params, k)
        n_mej = mejora_en_n_metricas(met, previa) if previa else np.nan
        fila = {"modelo": nombre, "n_variables": len(vars_act), "mejora_en_n_de_5_metricas": n_mej}
        fila.update({m: met[m] for m in C.COLS_METRICAS})
        fila.update({f"{m}_art": v for m, v in zip(C.COLS_METRICAS, C.REPORTADO_TABLA3[nombre])})
        filas.append(fila)
        previa = met
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ Modelo final + SHAP (Tabla 5, Figs 3-7)
def modelo_final_shap(df, n_shap=6000, semilla=C.SEMILLA):
    """Modelo final con partición simple 80/20 y TreeSHAP sobre una muestra de validación."""
    modelo, tr, te, met = holdout_lightgbm(df, C.VARS_FINAL)
    X_te = df.iloc[te][C.VARS_FINAL]
    rng = np.random.default_rng(semilla)
    sub = rng.choice(len(X_te), size=min(n_shap, len(X_te)), replace=False)
    Xs = X_te.iloc[sub]
    shap_vals = shap.TreeExplainer(modelo).shap_values(Xs)
    imp = pd.Series(np.abs(shap_vals).mean(0), index=C.VARS_FINAL).sort_values(ascending=False)
    return {"modelo": modelo, "tr": tr, "te": te, "metricas": met, "Xs": Xs, "shap": shap_vals, "importancia": imp}


# ------------------------------------------------------------------ OLS vs SAR (Tablas 6-10)
def comparar_econometricos(df, permutaciones=199):
    """Estima OLS (HC1) y SAR (2SLS) en toda la muestra, Moran e importancias; CV 5-fold de ambos."""
    xy = lat_lon_a_km(df.latitude.values, df.longitude.values)
    W = construir_W_knn(xy, C.K_VECINOS)
    y = df["ln_price"].values
    X = df[VARS_ECON].values.astype(float)
    ols = ols_fit(df)
    sar = sar_2sls(y, X, W)
    mor = {"ln(price)": moran_I(y, W, permutaciones),
           "residuos OLS": moran_I(np.asarray(ols.resid), W, permutaciones),
           "residuos SAR": moran_I(sar["resid"], W, permutaciones)}
    coef = pd.DataFrame({"variable": ["constant"] + VARS_ECON,
                         "OLS_est": ols.params.values, "SAR_est": sar["coef"],
                         "OLS_art": [C.REPORTADO_COEF[v][0] for v in ["constant"] + VARS_ECON],
                         "SAR_art": [C.REPORTADO_COEF[v][1] for v in ["constant"] + VARS_ECON]})
    cv = cv_ols_sar(df, VARS_ECON, W=W)
    sd = df[VARS_ECON].std().values
    imp_ols = pd.Series(np.abs(ols.params.values[1:]) * sd, index=VARS_ECON).sort_values(ascending=False)
    imp_sar = pd.Series(np.abs(sar["coef"][1:]) * sd, index=VARS_ECON).sort_values(ascending=False)
    return {"W": W, "ols": ols, "sar": sar, "moran": mor, "coef": coef, "cv": cv,
            "imp_ols": imp_ols, "imp_sar": imp_sar}


def tabla_ranking(imp_shap, imp_ols, imp_sar, top=15):
    """Análogo a la Tabla 10 (sin latitud/longitud, como en el artículo)."""
    s = imp_shap.drop(["latitude", "longitude"])
    return pd.DataFrame({"pos": range(1, top + 1), "LightGBM+SHAP": s.index[:top],
                         "OLS": imp_ols.index[:top], "SAR": imp_sar.index[:top]})


def tabla_comparacion_resultados(resultados):
    """Resumen 'artículo vs. conjuntos analizados' de los resultados centrales (Tablas 5, 6, 7, 8, 9).

    `resultados`: dict {nombre del conjunto: (res_final, res_econ)}, p. ej. {"real": (res_final, res_econ)}.
    Devuelve las columnas: resultado, métrica, artículo, <un valor por conjunto>, <diferencia con el artículo por conjunto>."""
    f = []
    def agregar(res, met, art, fn):
        f.append({"resultado": res, "métrica": met, "artículo": art, **{n: fn(r) for n, r in resultados.items()}})
    for m, v in zip(C.COLS_METRICAS, C.REPORTADO_TABLA5):
        agregar("Tabla 5 LightGBM holdout 80/20", m, v, lambda r, m=m: r[0]["metricas"][m])
    for modelo in ["OLS", "SAR"]:
        for m, v in zip(C.COLS_METRICAS, C.REPORTADO_TABLA9[modelo]):
            agregar(f"Tabla 9 {modelo} 5-fold CV", m, v, lambda r, m=m, modelo=modelo: r[1]["cv"][modelo][m])
    for k, v in C.REPORTADO_MORAN.items():
        agregar("Tabla 6-7 I de Moran", k, v, lambda r, k=k: r[1]["moran"][k][0])
    agregar("Tabla 8 R2 en muestra (log precio)", "OLS", C.REPORTADO_R2_INSAMPLE["OLS"], lambda r: r[1]["ols"].rsquared)
    agregar("Tabla 8 R2 en muestra (log precio)", "SAR", C.REPORTADO_R2_INSAMPLE["SAR"], lambda r: r[1]["sar"]["R2_insample"])
    agregar("Tabla 8 parámetro autorregresivo", "rho", C.RHO_REPORTADO, lambda r: r[1]["sar"]["rho"])
    out = pd.DataFrame(f)
    for n in resultados:
        out[f"dif_{n}"] = out[n] - out["artículo"]
    return out


# ------------------------------------------------------------------ Extensión: tamaño de muestra
def evaluar_tamano(df, n, semilla, k=5):
    """Mismo procedimiento del artículo sobre una submuestra aleatoria de tamaño n.

    k_vecinos de W se escala con n para conservar el MISMO radio espacial (700 vecinos sobre 52.039)."""
    sub = df.sample(n=n, random_state=semilla).reset_index(drop=True) if n < len(df) else df.reset_index(drop=True)
    k_w = max(8, int(round(C.K_VECINOS * len(sub) / C.N_FINAL)))
    met_l, oof_l = cv_lightgbm(sub, C.VARS_FINAL, PARAMS_LGBM, k)
    cv, oof_e = cv_ols_sar(sub, VARS_ECON, k_vecinos=k_w, k=k, devolver_oof=True)
    y_uf = np.expm1(sub.ln_price.values)
    filas = []
    for modelo, met, oof in [("LightGBM", met_l, oof_l), ("OLS", cv["OLS"], oof_e["OLS"]), ("SAR", cv["SAR"], oof_e["SAR"])]:
        fila = {"n": len(sub), "semilla": semilla, "modelo": modelo, "k_vecinos_W": k_w}
        fila.update({m: met[m] for m in C.COLS_METRICAS})
        fila.update(metricas_razon(y_uf, np.expm1(oof)))      # complemento OE4 (no está en el artículo)
        filas.append(fila)
    return filas
