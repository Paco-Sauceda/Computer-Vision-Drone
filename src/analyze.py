"""
analyze.py — detections.json → estadísticas y gráficas.

Cruza la tasa de detección y la confianza contra las condiciones ópticas del
frame (brillo, nitidez, % de highlights quemados) para encontrar patrones de
falla.

Punto metodológico central: las métricas ópticas NO son comparables entre
clips. La varianza del laplaciano depende de resolución, lente, escena y hora;
el brillo medio depende de la escena. Si se agregan los 10 clips en un solo
`qcut`, los bins terminan separando clips, no condiciones ópticas — y el
resultado es una paradoja de Simpson. Por eso cada gráfica se produce en dos
versiones: agregada (cruda) y normalizada dentro de cada clip (z-score por
`video`). La segunda es la que sostiene cualquier afirmación causal.

Uso:
    python src/analyze.py
    python src/analyze.py --detections output/detections.json --out output --bins 6
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLOR_BARRA = "#4C72B0"
COLOR_BARRA_CLIP = "#C44E52"

# Columnas ópticas que se analizan contra la tasa de detección.
METRICAS = ("brillo_medio", "nitidez", "pct_quemado")


def cargar_dataframes(detections_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (df_frames, df_detecciones).

    df_frames: un renglón por frame, con métricas ópticas y n_detecciones.
    df_detecciones: un renglón por detección individual, con clase y confianza,
    más las métricas ópticas del frame al que pertenece (para cruzar por clase).
    """
    data = json.loads(detections_path.read_text(encoding="utf-8"))

    filas_frame = []
    filas_deteccion = []

    for r in data["resultados"]:
        base = {
            "frame": r["frame"],
            "video": r.get("video", "unico"),
            "segundo": r.get("segundo"),
            "brillo_medio": r.get("brillo_medio"),
            "nitidez": r.get("nitidez"),
            "pct_quemado": r.get("pct_quemado"),
            "n_detecciones": r["n_detecciones"],
            "confianza_media": r["confianza_media"],
        }
        filas_frame.append(base)

        for det in r["detecciones"]:
            filas_deteccion.append(
                {
                    **base,
                    "clase": det["clase"],
                    "confianza": det["confianza"],
                    "area_px": det["area_px"],
                }
            )

    return pd.DataFrame(filas_frame), pd.DataFrame(filas_deteccion)


def resumen_por_clase(df_det: pd.DataFrame) -> pd.DataFrame:
    return (
        df_det.groupby("clase")
        .agg(
            n_detecciones=("clase", "count"),
            confianza_media=("confianza", "mean"),
            confianza_std=("confianza", "std"),
        )
        .round(3)
        .sort_values("n_detecciones", ascending=False)
    )


def normalizar_por_clip(df: pd.DataFrame, columna: str) -> pd.Series:
    """Z-score de `columna` dentro de cada clip (`video`).

    Responde a la limitación que el propio README declara: solo tiene sentido
    comparar frames del mismo clip. Normalizar dentro del clip mide lo que la
    métrica dice medir — variación óptica *dentro* de una misma toma — en vez
    de medir "de qué clip viene este frame".
    """
    g = df.groupby("video")[columna]
    return (df[columna] - g.transform("mean")) / g.transform("std")


def wilson(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción, en %.

    Con ~100 frames por bin, una tasa del 65% tiene un IC95 de ±9 puntos. Sin
    esto, bins de 33% y 45% se leen como si fueran distintos cuando no lo son.
    """
    if n == 0:
        return (0.0, 0.0)
    p = exitos / n
    denom = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denom
    margen = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centro - margen) * 100, min(1.0, centro + margen) * 100)


def tabla_tasa_por_bin(
    df_frames: pd.DataFrame, columna: str, bins: int = 6, por_clip: bool = False
) -> pd.DataFrame:
    """Tasa de detección por bin de cuantiles, con n e IC95 de Wilson.

    Usa cuantiles y no bins de ancho fijo para que cada barra tenga una muestra
    comparable detrás. Con anchos fijos, los extremos del rango caían en bins
    de 1-2 frames y una sola detección se veía como 100% — ruido, no señal.
    """
    df = df_frames.copy()
    df["tiene_deteccion"] = df["n_detecciones"] > 0

    col = columna
    if por_clip:
        col = f"{columna}_z"
        df[col] = normalizar_por_clip(df, columna)
        df = df.dropna(subset=[col])

    df["bin"] = pd.qcut(df[col], q=bins, duplicates="drop")
    agg = df.groupby("bin", observed=True).agg(
        frames=("frame", "count"),
        con_deteccion=("tiene_deteccion", "sum"),
    )
    agg["tasa_deteccion_pct"] = (agg["con_deteccion"] / agg["frames"] * 100).round(1)
    ics = [wilson(int(e), int(n)) for e, n in zip(agg["con_deteccion"], agg["frames"])]
    agg["ic95_bajo"] = [round(lo, 1) for lo, _ in ics]
    agg["ic95_alto"] = [round(hi, 1) for _, hi in ics]
    agg.index = agg.index.astype(str)
    return agg


def graficar_deteccion_vs(
    df_frames: pd.DataFrame,
    columna: str,
    out_path: Path,
    bins: int = 6,
    por_clip: bool = False,
) -> pd.DataFrame:
    """Barras de tasa de detección por bin, con n visible y barras de error IC95."""
    tabla = tabla_tasa_por_bin(df_frames, columna, bins=bins, por_clip=por_clip)

    tasa = tabla["tasa_deteccion_pct"]
    err_bajo = tasa - tabla["ic95_bajo"]
    err_alto = tabla["ic95_alto"] - tasa

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(
        range(len(tasa)),
        tasa.values,
        color=COLOR_BARRA_CLIP if por_clip else COLOR_BARRA,
        yerr=[err_bajo.values, err_alto.values],
        capsize=4,
        ecolor="#444444",
    )
    for i, (valor, n) in enumerate(zip(tasa.values, tabla["frames"].values)):
        ax.annotate(f"n={n}", (i, 2), ha="center", va="bottom", fontsize=8, color="white")

    ax.set_xticks(range(len(tasa)))
    ax.set_xticklabels(tabla.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% de frames con al menos una detección")
    ax.set_xlabel(f"{columna} — bin de cuantiles" + (" (z dentro del clip)" if por_clip else ""))
    sufijo = " — normalizado dentro de cada clip" if por_clip else " — agregado sobre todos los clips"
    ax.set_title(f"Tasa de detección por {columna}{sufijo}\nbarras de error: IC95 (Wilson)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return tabla


def graficar_confianza_vs(df_det: pd.DataFrame, columna: str, out_path: Path) -> None:
    """Scatter exploratorio de confianza por detección.

    Aviso: hay pseudorreplicación. Un frame con 8 coches aporta 8 puntos con la
    misma métrica óptica, así que los frames con mucho contenido dominan la
    nube. Sirve para mirar, no para calcular una correlación.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_det[columna], df_det["confianza"], alpha=0.4, s=20)
    ax.set_xlabel(columna)
    ax.set_ylabel("confianza de detección")
    ax.set_title(f"Confianza vs {columna}  (exploratorio — puntos no independientes)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def tabla_correlaciones(df_frames: pd.DataFrame) -> pd.DataFrame:
    """Correlación de cada métrica con la detección: agregada vs dentro de clip.

    Es la tabla que decide si un hallazgo es real o es un efecto de clip. Si el
    valor agregado y el de dentro de clip no coinciden en signo y magnitud, el
    patrón agregado es confounding, no física.
    """
    df = df_frames.copy()
    df["det"] = (df["n_detecciones"] > 0).astype(int)

    filas = []
    for col in METRICAS:
        z = normalizar_por_clip(df, col)
        filas.append(
            {
                "metrica": col,
                "pearson_agregado": round(float(df[col].corr(df["det"])), 3),
                "pearson_dentro_de_clip": round(float(z.corr(df["det"])), 3),
                "spearman_agregado": round(float(df[col].corr(df["det"], method="spearman")), 3),
                "spearman_dentro_de_clip": round(float(z.corr(df["det"], method="spearman")), 3),
            }
        )
    return pd.DataFrame(filas).set_index("metrica")


def tabla_por_clip(df_frames: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por clip. La unidad experimental real son 10 clips, no 603 frames."""
    df = df_frames.copy()
    df["tiene_deteccion"] = df["n_detecciones"] > 0
    return (
        df.groupby("video")
        .agg(
            frames=("frame", "count"),
            detecciones=("n_detecciones", "sum"),
            tasa_deteccion_pct=("tiene_deteccion", lambda s: round(s.mean() * 100, 1)),
            brillo_medio=("brillo_medio", lambda s: round(s.mean(), 1)),
            nitidez_media=("nitidez", lambda s: round(s.mean(), 1)),
            quemado_medio_pct=("pct_quemado", lambda s: round(s.mean(), 2)),
        )
        .sort_index()
    )


def main():
    p = argparse.ArgumentParser(description="Analiza detections.json y genera gráficas.")
    p.add_argument("--detections", type=Path, default=Path("output/detections.json"))
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--bins", type=int, default=6, help="Número de cuantiles por gráfica (default: 6)")
    args = p.parse_args()

    if not args.detections.exists():
        raise SystemExit(f"No existe: {args.detections}. ¿Corriste detect.py primero?")

    df_frames, df_det = cargar_dataframes(args.detections)

    charts_dir = args.out / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    resumen = resumen_por_clase(df_det)
    resumen.to_csv(args.out / "stats_por_clase.csv")
    df_frames.to_csv(args.out / "stats_frames.csv", index=False)

    correl = tabla_correlaciones(df_frames)
    correl.to_csv(args.out / "stats_correlaciones.csv")

    tabla_por_clip(df_frames).to_csv(args.out / "stats_por_clip.csv")

    nombres = {"brillo_medio": "brillo", "nitidez": "nitidez", "pct_quemado": "quemado"}
    for col in METRICAS:
        corto = nombres[col]
        graficar_confianza_vs(df_det, col, charts_dir / f"confianza_vs_{corto}.png")
        graficar_deteccion_vs(df_frames, col, charts_dir / f"tasa_deteccion_vs_{corto}.png", bins=args.bins)
        graficar_deteccion_vs(
            df_frames,
            col,
            charts_dir / f"tasa_deteccion_vs_{corto}_por_clip.png",
            bins=args.bins,
            por_clip=True,
        )

    print("Resumen por clase:")
    print(resumen.to_string())
    print()
    print("Correlación con la tasa de detección (agregado vs dentro de clip):")
    print(correl.to_string())
    print()
    print("Si el signo cambia entre las dos columnas, el patrón agregado es un efecto de clip.")
    print()
    print(f"CSV frames        : {args.out / 'stats_frames.csv'}")
    print(f"CSV clases        : {args.out / 'stats_por_clase.csv'}")
    print(f"CSV correlaciones : {args.out / 'stats_correlaciones.csv'}")
    print(f"CSV por clip      : {args.out / 'stats_por_clip.csv'}")
    print(f"Gráficas          : {charts_dir}")


if __name__ == "__main__":
    main()
