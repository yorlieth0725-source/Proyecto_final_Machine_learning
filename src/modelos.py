"""Modelos y métricas del artículo: LightGBM (+SHAP), OLS con errores robustos y SAR.

Métricas (Tabla 2): MAPE, MdAPE, MAE, RMSE y R2, calculadas en UF (tras expm1() de las predicciones; la variable dependiente es log1p(precio), como en el código de los autores).
Se agregan, como COMPLEMENTO para el trabajo de grado (no están en el artículo): mediana de la
razón, COD y PRD, que son las métricas del OE4 (estándares de razón de avalúo).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.model_selection import KFold

from . import config as C
from .espacial import construir_W_knn, lat_lon_a_km, resolver_sar

# Hiperparámetros del código de los autores (lgbm_section.ipynb), salvo que aquí se usa GBDT con 400 iteraciones
# en lugar de DART con 8.000 (DART tarda horas y con pocas iteraciones no converge).
PARAMS_LGBM = dict(
    boosting_type="gbdt", objective="regression", n_estimators=400, learning_rate=0.1,
    colsample_bytree=0.9, subsample=0.7, subsample_freq=10,
    max_depth=40, num_leaves=256, max_bin=512, min_child_samples=10,
    random_state=C.SEMILLA, n_jobs=-1, verbose=-1,
)
# Hiperparámetros originales completos (DART, 8.000 iteraciones): horas de cómputo; ver notebooks/colab.
PARAMS_DART = dict(PARAMS_LGBM, boosting_type="dart", n_estimators=8000, drop_rate=0.12)


# --------------------------------------------------------------------------- métricas
def metricas(y_uf, yhat_uf):
    """MAPE %, MdAPE %, MAE, RMSE (UF) y R2 (en UF), como la Tabla 2 del artículo."""
    y, f = np.asarray(y_uf, float), np.asarray(yhat_uf, float)
    ape = np.abs((y - f) / y)
    return {"MAPE": 100 * ape.mean(), "MdAPE": 100 * np.median(ape), "MAE": np.abs(y - f).mean(),
            "RMSE": np.sqrt(((y - f) ** 2).mean()), "R2": 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()}


def metricas_razon(y_uf, yhat_uf):
    """Métricas de razón de avalúo (complemento OE4): razón = predicho / observado."""
    r = np.asarray(yhat_uf, float) / np.asarray(y_uf, float)
    med = np.median(r)
    cod = 100 * np.mean(np.abs(r - med)) / med
    prd = r.mean() / (np.sum(yhat_uf) / np.sum(y_uf))
    return {"mediana_razon": med, "COD": cod, "PRD": prd}


def mejora_en_n_metricas(nueva, previa):
    """Criterio del artículo: se conserva el cambio si mejora >= 3 de las 5 métricas."""
    mejoras = [nueva["MAPE"] < previa["MAPE"], nueva["MdAPE"] < previa["MdAPE"], nueva["MAE"] < previa["MAE"],
               nueva["RMSE"] < previa["RMSE"], nueva["R2"] > previa["R2"]]
    return sum(mejoras)


# --------------------------------------------------------------------------- LightGBM
def _lgbm(params):
    return lgb.LGBMRegressor(**params)


def cv_lightgbm(df, variables, params=None, k=5, semilla=C.SEMILLA_CV):
    """5-fold CV aleatorio (80/20) con objetivo ln(precio). Devuelve métricas promediadas en los folds
    (calculadas en UF) y las predicciones fuera de muestra."""
    params = params or PARAMS_LGBM
    X, y = df[variables].values, df["ln_price"].values
    oof = np.zeros(len(df))
    filas = []
    for tr, te in KFold(k, shuffle=True, random_state=semilla).split(X):
        m = _lgbm(params).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        filas.append(metricas(np.expm1(y[te]), np.expm1(oof[te])))
    prom = pd.DataFrame(filas).mean().to_dict()
    return prom, oof


def holdout_lightgbm(df, variables, params=None, semilla=C.SEMILLA, test_size=0.2):
    """Partición simple 80/20 (para SHAP, Tabla 5). Si el DataFrame trae la columna `set` (datos reales de
    Mendeley: Training/Test) se usa esa partición, la misma de los autores; si no, una partición aleatoria.
    Devuelve modelo, posiciones de entrenamiento y de prueba, y métricas."""
    params = params or PARAMS_LGBM
    if "set" in df.columns:
        es_te = (df["set"].values == "Test")
        tr, te = np.where(~es_te)[0], np.where(es_te)[0]
    else:
        rng = np.random.default_rng(semilla)
        idx = rng.permutation(len(df))
        n_te = int(len(df) * test_size)
        te, tr = idx[:n_te], idx[n_te:]
    m = _lgbm(params).fit(df.iloc[tr][variables].values, df.iloc[tr]["ln_price"].values)
    pred = m.predict(df.iloc[te][variables].values)
    return m, tr, te, metricas(np.expm1(df.iloc[te]["ln_price"].values), np.expm1(pred))


# --------------------------------------------------------------------------- OLS y SAR
VARS_ECON = [v for v in C.VARS_FINAL if v not in ("latitude", "longitude")]   # 32 variables (sin lat/lon)


def ols_fit(df, variables=VARS_ECON):
    """OLS con errores robustos HC1 sobre ln(precio) (Eq. 3)."""
    X = sm.add_constant(df[variables].astype(float))
    return sm.OLS(df["ln_price"].values, X).fit(cov_type="HC1")


def sar_2sls(y, X, W, instrumentos_orden=2):
    """SAR (Eq. 6) estimado por mínimos cuadrados en dos etapas espaciales (Kelejian-Prucha).

    y = rho W y + X b + e. Instrumentos: [X, WX, W^2 X]. Es el mismo enfoque del código de los
    autores (spreg.GM_Lag, 2SLS), que usa solo WX como instrumentos; aquí se agrega W²X.
    Devuelve dict con rho, coeficientes (const + variables), errores estándar y R2 en la muestra.
    """
    n = len(y)
    X1 = np.column_stack([np.ones(n), X])
    Wy = W @ y
    Z = [X1]
    WX = W @ X
    Z.append(WX)
    if instrumentos_orden >= 2:
        Z.append(W @ WX)
    Z = np.column_stack(Z)
    # primera etapa: Wy sobre instrumentos
    g, *_ = np.linalg.lstsq(Z, Wy, rcond=None)
    Wy_hat = Z @ g
    # segunda etapa
    R = np.column_stack([Wy_hat, X1])
    b, *_ = np.linalg.lstsq(R, y, rcond=None)
    # errores estándar 2SLS: residuos con Wy observado
    Ro = np.column_stack([Wy, X1])
    e = y - Ro @ b
    s2 = (e @ e) / (n - R.shape[1])
    cov = s2 * np.linalg.inv(R.T @ R)
    se = np.sqrt(np.diag(cov))
    yhat_ins = Ro @ b
    r2 = 1 - ((y - yhat_ins) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"rho": b[0], "coef": b[1:], "se": se[1:], "se_rho": se[0], "R2_insample": r2, "resid": e}


def prediccion_forma_reducida(df_all, variables, rho, coef, W_all):
    """Predicción SAR sin usar y de los puntos evaluados: yhat = (I - rho W)^-1 (a + X b).
    W_all se construye con TODOS los puntos (sólo usa coordenadas), como `predy_e` de spreg."""
    X1 = np.column_stack([np.ones(len(df_all)), df_all[variables].values.astype(float)])
    return resolver_sar(W_all, X1 @ coef, rho)


def cv_ols_sar(df, variables=VARS_ECON, k_vecinos=C.K_VECINOS, k=5, semilla=C.SEMILLA_CV, W=None, devolver_oof=False):
    """5-fold CV de OLS y SAR con las mismas variables (sin lat/lon). Devuelve métricas por modelo."""
    xy = lat_lon_a_km(df.latitude.values, df.longitude.values)
    W = W if W is not None else construir_W_knn(xy, k_vecinos)
    y = df["ln_price"].values
    X = df[variables].values.astype(float)
    X1 = np.column_stack([np.ones(len(df)), X])
    res = {"OLS": [], "SAR": []}
    oof = {"OLS": np.zeros(len(df)), "SAR": np.zeros(len(df))}
    for tr, te in KFold(k, shuffle=True, random_state=semilla).split(X):
        # OLS
        b, *_ = np.linalg.lstsq(X1[tr], y[tr], rcond=None)
        oof["OLS"][te] = X1[te] @ b
        res["OLS"].append(metricas(np.expm1(y[te]), np.expm1(X1[te] @ b)))
        # SAR: W de entrenamiento (sólo puntos de entrenamiento) para estimar
        W_tr = construir_W_knn(xy[tr], k_vecinos)
        fit = sar_2sls(y[tr], X[tr], W_tr)
        coef = np.r_[fit["coef"]]
        yhat_all = resolver_sar(W, X1 @ coef, fit["rho"])
        oof["SAR"][te] = yhat_all[te]
        res["SAR"].append(metricas(np.expm1(y[te]), np.expm1(yhat_all[te])))
    out = {m: pd.DataFrame(v).mean().to_dict() for m, v in res.items()}
    return (out, oof) if devolver_oof else out


def cv_sar_vecinos(df, variables=VARS_ECON, k_vecinos=C.K_VECINOS, k=5, semilla=C.SEMILLA_CV):
    """Variante de predicción del SAR: yhat = X b + rho * (promedio del ln(precio) de los vecinos de ENTRENAMIENTO más cercanos).

    A diferencia de la forma reducida (que sólo usa X y coordenadas), usa los precios observados de los vecinos, como haría un
    avalúo por comparables. Es una elección de este proyecto: el artículo no detalla cómo predice el SAR fuera de muestra.
    Los vecinos de cada punto de validación son los k_vecinos*(n_entrenamiento/n) puntos de entrenamiento más cercanos."""
    from scipy.spatial import cKDTree
    xy = lat_lon_a_km(df.latitude.values, df.longitude.values)
    y = df["ln_price"].values
    X = df[variables].values.astype(float)
    X1 = np.column_stack([np.ones(len(df)), X])
    res = []
    for tr, te in KFold(k, shuffle=True, random_state=semilla).split(X):
        fit = sar_2sls(y[tr], X[tr], construir_W_knn(xy[tr], k_vecinos))
        k_w = max(8, int(round(k_vecinos * len(tr) / len(df))))
        idx = cKDTree(xy[tr]).query(xy[te], k=k_w, workers=-1)[1]
        yhat = X1[te] @ np.r_[fit["coef"]] + fit["rho"] * y[tr][idx].mean(1)
        res.append(metricas(np.expm1(y[te]), np.expm1(yhat)))
    return pd.DataFrame(res).mean().to_dict()
