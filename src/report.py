"""
report.py — detections.json → reporte markdown legible.

No inventa nada nuevo: reutiliza las funciones de analyze.py para no duplicar
la lógica de agregación, y arma un documento de una sola pieza con las tablas,
el desglose por clip y las gráficas ya generadas en output/charts/.

Cada métrica se reporta dos veces: agregada sobre todos los clips y
normalizada dentro de cada clip. La comparación entre las dos es el contenido
real del reporte — si difieren, el patrón agregado es un efecto de clip.

Uso:
    python src/report.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze import (  # noqa: E402
    METRICAS,
    cargar_dataframes,
    resumen_por_clase,
    tabla_correlaciones,
    tabla_por_clip,
    tabla_tasa_por_bin,
)

TITULOS = {
    "brillo_medio": "brillo (media del canal V)",
    "nitidez": "nitidez (varianza del laplaciano)",
    "pct_quemado": "% de píxeles quemados",
}
ARCHIVOS = {"brillo_medio": "brillo", "nitidez": "nitidez", "pct_quemado": "quemado"}


def a_markdown_tabla(df: pd.DataFrame, index_nombre: str) -> str:
    nombre_indice = df.index.name or "index"
    return df.reset_index().rename(columns={nombre_indice: index_nombre}).to_markdown(index=False)


def generar_reporte(detections_path: Path, out_path: Path, bins: int = 6) -> str:
    meta = json.loads(detections_path.read_text(encoding="utf-8"))
    df_frames, df_det = cargar_dataframes(detections_path)

    n_clips = meta.get("n_videos", df_frames["video"].nunique())
    tasa_global = round((df_frames["n_detecciones"] > 0).mean() * 100, 1)
    fps_extraccion = meta.get("fps_extraccion", "no registrado")

    partes = [
        "# Reporte — drone-vision",
        "",
        f"Modelo: `{meta['modelo']}` · umbral de confianza: `{meta['umbral_conf']}`"
        f" · imgsz: `{meta.get('imgsz', 'no registrado')}`"
        f" · ultralytics: `{meta.get('ultralytics_version', 'no registrada')}`",
        "",
        f"- Clips analizados: **{n_clips}**",
        f"- Frames muestreados: **{meta['n_frames']}** (fps de extracción: {fps_extraccion})",
        f"- Detecciones totales: **{meta['n_detecciones_total']}**",
        f"- Frames con al menos una detección: **{tasa_global}%**",
        f"- Frames sin ninguna detección: **{meta['frames_sin_deteccion']}** "
        f"({100 - tasa_global:.1f}% del total)",
        "",
        "> **La unidad experimental son los clips, no los frames.** Hay "
        f"{meta['n_frames']} observaciones pero solo {n_clips} tomas independientes. "
        "Cualquier patrón que se sostenga en el agregado y desaparezca al normalizar "
        "dentro del clip es un efecto de clip, no una relación óptica.",
        "",
        "## Correlación con la tasa de detección",
        "",
        "Agregado sobre todos los clips vs. normalizado dentro de cada clip "
        "(z-score por `video`). Si el signo cambia entre las dos columnas, el "
        "patrón agregado es una paradoja de Simpson.",
        "",
        a_markdown_tabla(tabla_correlaciones(df_frames), "métrica"),
        "",
        "## Por clip",
        "",
        a_markdown_tabla(tabla_por_clip(df_frames), "clip"),
        "",
        "## Por clase detectada",
        "",
        a_markdown_tabla(resumen_por_clase(df_det), "clase"),
        "",
    ]

    for col in METRICAS:
        corto = ARCHIVOS[col]
        partes += [
            f"## Tasa de detección por {TITULOS[col]}",
            "",
            "**Agregado sobre todos los clips** (IC95 de Wilson):",
            "",
            a_markdown_tabla(tabla_tasa_por_bin(df_frames, col, bins=bins), f"{col} (bin)"),
            "",
            f"![](charts/tasa_deteccion_vs_{corto}.png)",
            "",
            "**Normalizado dentro de cada clip** (z-score por clip):",
            "",
            a_markdown_tabla(
                tabla_tasa_por_bin(df_frames, col, bins=bins, por_clip=True), f"{col} z (bin)"
            ),
            "",
            f"![](charts/tasa_deteccion_vs_{corto}_por_clip.png)",
            "",
        ]

    texto = "\n".join(partes) + "\n"
    out_path.write_text(texto, encoding="utf-8")
    return texto


def main():
    p = argparse.ArgumentParser(description="Genera output/report.md desde detections.json.")
    p.add_argument("--detections", type=Path, default=Path("output/detections.json"))
    p.add_argument("--out", type=Path, default=Path("output/report.md"))
    p.add_argument("--bins", type=int, default=6)
    args = p.parse_args()

    if not args.detections.exists():
        raise SystemExit(f"No existe: {args.detections}. Corre extract.py + detect.py primero.")

    generar_reporte(args.detections, args.out, bins=args.bins)
    print(f"Reporte: {args.out}")


if __name__ == "__main__":
    main()
