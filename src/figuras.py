"""Figuras del proyecto (matplotlib). Paleta categórica validada: azul = LightGBM,
naranja = OLS, agua = SAR, gris = valor reportado por el artículo (referencia, no una serie)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from . import config as C

AZUL, NARANJA, AGUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66"
TXT, TXT2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRILLA, "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
    "text.color": TXT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _guardar(fig, nombre):
    C.DIR_FIGURAS.mkdir(exist_ok=True)
    ruta = C.DIR_FIGURAS / nombre
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


REAL = "#3b3a37"        # tinta oscura = resultado con los datos reales; gris claro = valor del artículo


def fig_distribuciones(df, nombre="real"):
    """Descripción del conjunto (precio, superficie, mapa de precio por m2)."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].hist(df.price_uf, bins=60, color=REAL, edgecolor="#fcfcfb", linewidth=0.4)
    ax[0].set_title("Precio publicado (UF)")
    ax[0].set_xlabel("UF")
    ax[0].set_ylabel("anuncios")
    ax[1].hist(df.total_surface.clip(upper=500), bins=60, color=REAL, edgecolor="#fcfcfb", linewidth=0.4)
    ax[1].set_title("Superficie total (m², recortada en 500)")
    ax[1].set_xlabel("m²")
    muestra = df.sample(15000, random_state=1)
    pm2 = muestra.price_uf / muestra.total_surface
    sc = ax[2].scatter(muestra.longitude, muestra.latitude, c=pm2.clip(upper=pm2.quantile(0.98)), s=2,
                       cmap="Blues", linewidths=0)
    ax[2].set_title("Precio por m² (UF/m²)")
    ax[2].set_xlabel("longitud"); ax[2].set_ylabel("latitud"); ax[2].grid(False)
    fig.colorbar(sc, ax=ax[2], shrink=0.8)
    fig.tight_layout()
    return _guardar(fig, f"01_distribuciones_{nombre}.png")


def fig_secuencia(tabs):
    """Etapa 6: MAPE y R2 de la secuencia de modelos. `tabs` = {"real": tabla} (con las columnas `_art` del artículo)."""
    nombres = list(tabs)
    base = tabs[nombres[0]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(base))
    etiq = ["Base", "+ICVU", "+amenid.", "+nuevo", "+SIMCE", "+colegios", "+distancias", "+fotos"]
    estilo = {"real": dict(color=AZUL, ls="-", marker="o", mfc=AZUL)}
    for a, m, tit in [(ax[0], "MAPE", "MAPE (%) en validación cruzada"), (ax[1], "R2", "R² (en UF) en validación cruzada")]:
        a.plot(x, base[f"{m}_art"], color=GRIS, lw=2, marker="o", ms=5, ls="--", label="Artículo")
        for n in nombres:
            a.plot(x, tabs[n][m], lw=2, ms=6, label="Datos reales", **estilo[n])
        a.set_xticks(x); a.set_xticklabels(etiq, rotation=30, ha="right")
        a.set_title(tit); a.margins(y=0.14)
        a.legend(loc="upper right" if m == "MAPE" else "lower right", fontsize=8)
    fig.tight_layout()
    return _guardar(fig, "02_secuencia_modelos_vs_articulo.png")


def fig_shap_beeswarm(res, nombre="real"):
    shap.summary_plot(res["shap"], res["Xs"], max_display=20, show=False, plot_size=(8, 7))
    plt.title("SHAP: impacto en ln(precio) (20 variables más importantes)", loc="left", fontsize=11, fontweight="bold")
    fig = plt.gcf()
    return _guardar(fig, f"03_shap_beeswarm_{nombre}.png")


def fig_shap_dependencia(res, nombre="real"):
    """Dependencias análogas a las Figs 5-6 del artículo."""
    vars_ = ["total_surface", "quality_municipality", "simce_4b", "bathrooms", "nearby_schools_3km", "dist_subway"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    cols = {"total_surface": "quality_municipality", "quality_municipality": "simce_4b", "simce_4b": "quality_municipality",
            "bathrooms": "quality_municipality", "nearby_schools_3km": "simce_4b", "dist_subway": None}
    for a, v in zip(ax.ravel(), vars_):
        shap.dependence_plot(v, res["shap"], res["Xs"], interaction_index=cols[v], ax=a, show=False,
                             dot_size=6, alpha=0.6)
        a.set_title(v)
    fig.tight_layout()
    return _guardar(fig, f"04_shap_dependencia_{nombre}.png")


def fig_comparacion_modelos(comp):
    """Etapa 6: MAPE, MdAPE y R2 de LightGBM, OLS y SAR; artículo (gris) y datos reales (color del modelo)."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    modelos = [("LightGBM", AZUL, "Tabla 5 LightGBM holdout 80/20"), ("OLS", NARANJA, "Tabla 9 OLS 5-fold CV"),
               ("SAR", AGUA, "Tabla 9 SAR 5-fold CV")]
    for a, met, tit in zip(ax, ["MAPE", "MdAPE", "R2"], ["MAPE (%)", "MdAPE (%)", "R² (en UF)"]):
        for i, (nom, col, clave) in enumerate(modelos):
            fila = comp[(comp.resultado == clave) & (comp["métrica"] == met)].iloc[0]
            for dx, v, kw in [(-0.2, fila["artículo"], dict(color="#c9c8c3")), (0.2, fila["real"], dict(color=col))]:
                a.bar(i + dx, v, 0.37, edgecolor="none", **kw)
                a.text(i + dx, v, f"{v:.1f}" if met != "R2" else f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TXT2)
        a.set_xticks(range(3)); a.set_xticklabels([m[0] for m in modelos]); a.set_title(tit); a.grid(axis="x", visible=False)
        a.margins(y=0.12)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color="#c9c8c3", label="Artículo"), Patch(color=GRIS, label="Datos reales (color del modelo)")],
                 loc="upper left", fontsize=8)
    fig.tight_layout()
    return _guardar(fig, "05_comparacion_modelos.png")


def _etiq_n(a, ns):
    """Eje x con el tamaño de muestra en escala logarítmica y marcas en los n evaluados."""
    ns = sorted(ns)
    a.set_xscale("log")
    a.set_xticks(ns)
    a.set_xticklabels([f"{round(n/1000)}k" for n in ns], fontsize=8)
    a.minorticks_off()
    a.set_xlabel("Tamaño de muestra (anuncios)")


def fig_extension(res, nombre="real"):
    """Etapa 7: desempeño vs. tamaño de muestra (media y rango de las repeticiones) en un conjunto."""
    g = res.groupby(["modelo", "n"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    col = {"LightGBM": AZUL, "OLS": NARANJA, "SAR": AGUA}
    for a, met, tit in zip(ax, ["MAPE", "MdAPE", "R2"], ["MAPE (%)", "MdAPE (%)", "R² (en UF)"]):
        for modelo, c in col.items():
            s = g[met].agg(["mean", "min", "max"]).loc[modelo]
            a.plot(s.index, s["mean"], color=c, lw=2, marker="o", ms=6, label=modelo)
            a.fill_between(s.index, s["min"], s["max"], color=c, alpha=0.15, linewidth=0)
        a.set_title(tit); _etiq_n(a, res.n.unique())
    ax[0].legend(loc="center right")
    fig.suptitle(f"Datos {nombre}s".replace("reals", "reales"), x=0.01, ha="left", fontsize=10, color=TXT2)
    fig.tight_layout()
    return _guardar(fig, f"06_extension_tamano_muestra_{nombre}.png")


def fig_extension_razon(res, nombre="real"):
    """Etapa 7: métricas de razón (complemento para OE4) vs. tamaño de muestra."""
    g = res.groupby(["modelo", "n"])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.9))
    col = {"LightGBM": AZUL, "OLS": NARANJA, "SAR": AGUA}
    for a, met, tit in zip(ax, ["COD", "PRD"], ["COD (%)", "PRD"]):
        for modelo, c in col.items():
            s = g[met].mean().loc[modelo]
            a.plot(s.index, s.values, color=c, lw=2, marker="o", ms=6, label=modelo)
        a.set_title(tit); _etiq_n(a, res.n.unique())
    ax[1].axhspan(0.98, 1.03, color="#c9c8c3", alpha=0.35, lw=0)
    ax[1].text(1150, 0.984, "rango IAAO 0,98-1,03", fontsize=8, color=TXT2, va="bottom")
    ax[0].legend(loc="center right")
    fig.suptitle(f"Datos {nombre}s".replace("reals", "reales"), x=0.01, ha="left", fontsize=10, color=TXT2)
    fig.tight_layout()
    return _guardar(fig, f"07_extension_metricas_razon_{nombre}.png")
