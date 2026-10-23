"""Utilidades espaciales: matriz de pesos W (KNN), rezago espacial e I de Moran."""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def lat_lon_a_km(lat, lon, lat0=-33.45, lon0=-70.65):
    """Proyección plana local (km). Suficiente para distancias dentro de la región."""
    y = (np.asarray(lat) - lat0) * 111.0
    x = (np.asarray(lon) - lon0) * 111.0 * np.cos(np.radians(lat0))
    return np.column_stack([x, y])


def construir_W_knn(xy, k=700):
    """Matriz W binaria KNN (Eq. 4 del artículo) estandarizada por filas.

    Devuelve una matriz dispersa CSR n x n. Cada fila suma 1 (peso 1/k por vecino).
    """
    n = len(xy)
    k = min(k, n - 1)
    arbol = cKDTree(xy)
    _, idx = arbol.query(xy, k=k + 1, workers=-1)   # la primera columna es el propio punto
    idx = idx[:, 1:].astype(np.int32)
    filas = np.repeat(np.arange(n, dtype=np.int32), k)
    datos = np.full(n * k, 1.0 / k, dtype=np.float32)
    return sp.csr_matrix((datos, (filas, idx.ravel())), shape=(n, n))


def resolver_sar(W, z, rho, terminos=40):
    """Calcula y = (I - rho W)^-1 z con la serie de Neumann (rho<1, W estandarizada por filas)."""
    y = z.astype(np.float64).copy()
    termino = z.astype(np.float64).copy()
    for _ in range(terminos):
        termino = rho * (W @ termino)
        y += termino
        if np.abs(termino).max() < 1e-7:
            break
    return y


def moran_I(valores, W, permutaciones=199, semilla=0):
    """I de Moran global (Eq. 7) con W estandarizada por filas y p-valor por permutación."""
    z = np.asarray(valores, dtype=np.float64)
    z = z - z.mean()
    denom = float(z @ z)
    I = float(z @ (W @ z)) / denom          # con W por filas S0 = n, entonces n/S0 = 1
    if permutaciones <= 0:
        return I, np.nan
    rng = np.random.default_rng(semilla)
    mayores = 0
    for _ in range(permutaciones):
        zp = rng.permutation(z)
        mayores += (float(zp @ (W @ zp)) / denom) >= I
    p = (mayores + 1) / (permutaciones + 1)
    return I, p
