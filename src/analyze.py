"""
analyze.py — detections.json → estadísticas y gráficas.

Cruza la confianza de las detecciones contra las condiciones ópticas del frame
(brillo, nitidez, % de highlights quemados) para encontrar patrones de falla.
No hace ningún juicio por sí solo: deja los números y las gráficas, la
interpretación va en el README.

Uso:
    python src/analyze.py
    python src/analyze.py --detections output/detections.json --out output
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def graficar_confianza_vs(df_det: pd.DataFrame, columna: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_det[columna], df_det["confianza"], alpha=0.4, s=20)
    ax.set_xlabel(columna)
    ax.set_ylabel("confianza de detección")
    ax.set_title(f"Confianza vs {columna}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def graficar_deteccion_vs(df_frames: pd.DataFrame, columna: str, out_path: Path, bins: int = 6) -> None:
    """% de frames con al menos una detección, por bin (cuantiles) de la columna dada.

    Usa cuantiles, no bins de ancho fijo, para que cada barra tenga una muestra
    comparable detrás. Con anchos fijos, los extremos del rango caían en bins de
    1-2 frames y una sola detección se veía como 100% — ruido, no señal.
    """
    df = df_frames.copy()
    df["tiene_deteccion"] = df["n_detecciones"] > 0
    df["bin"] = pd.qcut(df[columna], q=bins, duplicates="drop")
    tasa = df.groupby("bin", observed=True)["tiene_deteccion"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    tasa.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("% de frames con al menos una detección")
    ax.set_xlabel(f"{columna} (bin)")
    ax.set_title(f"Tasa de detección por {columna}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Analiza detections.json y genera gráficas.")
    p.add_argument("--detections", type=Path, default=Path("output/detections.json"))
    p.add_argument("--out", type=Path, default=Path("output"))
    args = p.parse_args()

    if not args.detections.exists():
        raise SystemExit(f"No existe: {args.detections}. ¿Corriste detect.py primero?")

    df_frames, df_det = cargar_dataframes(args.detections)

    charts_dir = args.out / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    resumen = resumen_por_clase(df_det)
    resumen_path = args.out / "stats_por_clase.csv"
    resumen.to_csv(resumen_path)

    df_frames.to_csv(args.out / "stats_frames.csv", index=False)

    graficar_confianza_vs(df_det, "brillo_medio", charts_dir / "confianza_vs_brillo.png")
    graficar_confianza_vs(df_det, "nitidez", charts_dir / "confianza_vs_nitidez.png")
    graficar_confianza_vs(df_det, "pct_quemado", charts_dir / "confianza_vs_quemado.png")
    graficar_deteccion_vs(df_frames, "brillo_medio", charts_dir / "tasa_deteccion_vs_brillo.png")
    graficar_deteccion_vs(df_frames, "nitidez", charts_dir / "tasa_deteccion_vs_nitidez.png")

    print("Resumen por clase:")
    print(resumen.to_string())
    print()
    print(f"CSV frames : {args.out / 'stats_frames.csv'}")
    print(f"CSV clases : {resumen_path}")
    print(f"Gráficas   : {charts_dir}")


if __name__ == "__main__":
    main()
