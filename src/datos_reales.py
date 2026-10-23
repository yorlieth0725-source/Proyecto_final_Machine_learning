"""Datos reales del artículo (Mendeley Data, doi:10.17632/jdf7rzmnvw.1) y resumen de calibración.

`cargar_reales()` lee data/real/data_paper_plosone_set.csv (52.039 anuncios, columna `set` Training/Test,
mismo orden que data_paper_plosone.csv). Se agrega la variable dependiente como en el código de los
autores: ln_price = log1p(price).

"""
import numpy as np
import pandas as pd

from . import config as C

RUTA_REAL = C.DIR_DATOS / "real" / "data_paper_plosone_set.csv"


def cargar_reales(ruta=RUTA_REAL):
    df = pd.read_csv(ruta)
    df["price_uf"] = df["price"].astype(float)
    df["ln_price"] = np.log1p(df["price_uf"])
    return df
