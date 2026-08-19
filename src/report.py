"""
report.py — stats_frames.csv / detections.json → reporte markdown legible.

No inventa nada nuevo: reutiliza las mismas funciones de analyze.py para no
duplicar la lógica de agregación, y arma un documento de una sola pieza con
las tablas, el desglose por video y las gráficas ya generadas en output/charts/.

Uso:
    python src/report.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze import cargar_dataframes, resumen_por_clase  # noqa: E402


def tabla_por_video(df_frames: pd.DataFrame) -> pd.DataFrame:
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
        )
        .sort_index()
    )


def tabla_tasa_por_bin(df_frames: pd.DataFrame, columna: str, bins: int = 6) -> pd.DataFrame:
    df = df_frames.copy()
    df["tiene_deteccion"] = df["n_detecciones"] > 0
    df["bin"] = pd.qcut(df[columna], q=bins, duplicates="drop")
    tabla = df.groupby("bin", observed=True).agg(
        frames=("frame", "count"),
        tasa_deteccion_pct=("tiene_deteccion", lambda s: round(s.mean() * 100, 1)),
    )
    tabla.index = tabla.index.astype(str)
    return tabla


def a_markdown_tabla(df: pd.DataFrame, index_nombre: str) -> str:
    df = df.reset_index().rename(columns={df.index.name or "index": index_nombre})
    return df.to_markdown(index=False)


def generar_reporte(detections_path: Path, out_path: Path) -> str:
    meta = json.loads(detections_path.read_text(encoding="utf-8"))
    df_frames, df_det = cargar_dataframes(detections_path)

    n_videos = meta.get("n_videos", df_frames["video"].nunique())
    tasa_global = round((df_frames["n_detecciones"] > 0).mean() * 100, 1)

    partes = [
        "# Reporte — drone-vision",
        "",
        f"Modelo: `{meta['modelo']}` · umbral de confianza: `{meta['umbral_conf']}`",
        "",
        f"- Videos analizados: **{n_videos}**",
        f"- Frames muestreados (1fps): **{meta['n_frames']}**",
        f"- Detecciones totales: **{meta['n_detecciones_total']}**",
        f"- Frames sin ninguna detección: **{meta['frames_sin_deteccion']}** "
        f"({100 - tasa_global:.1f}% del total)",
        "",
        "## Por video",
        "",
        a_markdown_tabla(tabla_por_video(df_frames), "video"),
        "",
        "## Por clase detectada",
        "",
        a_markdown_tabla(resumen_por_clase(df_det), "clase"),
        "",
        "## Tasa de detección por brillo",
        "",
        a_markdown_tabla(tabla_tasa_por_bin(df_frames, "brillo_medio"), "brillo_medio (bin)"),
        "",
        "![](charts/tasa_deteccion_vs_brillo.png)",
        "",
        "## Tasa de detección por nitidez",
        "",
        a_markdown_tabla(tabla_tasa_por_bin(df_frames, "nitidez"), "nitidez (bin)"),
        "",
        "![](charts/tasa_deteccion_vs_nitidez.png)",
        "",
    ]

    texto = "\n".join(partes) + "\n"
    out_path.write_text(texto, encoding="utf-8")
    return texto


def main():
    p = argparse.ArgumentParser(description="Genera output/report.md desde detections.json.")
    p.add_argument("--detections", type=Path, default=Path("output/detections.json"))
    p.add_argument("--out", type=Path, default=Path("output/report.md"))
    args = p.parse_args()

    if not args.detections.exists():
        raise SystemExit(f"No existe: {args.detections}. Corre extract.py + detect.py primero.")

    generar_reporte(args.detections, args.out)
    print(f"Reporte: {args.out}")


if __name__ == "__main__":
    main()
