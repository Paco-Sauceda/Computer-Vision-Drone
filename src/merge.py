"""
merge.py — fusiona detections.json y frames_meta.json de varios videos en uno solo.

extract.py y detect.py procesan un video a la vez. Cuando el material son varios
clips (un vuelo completo dividido en tomas por el drone), cada uno se extrae y
detecta por separado y luego se fusiona aquí, para que analyze.py trabaje sobre
un único dataset con el nombre del video como columna.

Uso:
    for f in data/raw/*.MP4; do
        nombre=$(basename "${f%.*}")
        python src/extract.py "$f" --out "data/frames/$nombre" --meta "output/frames_meta_$nombre.json"
        python src/detect.py --frames "data/frames/$nombre" --meta "output/frames_meta_$nombre.json" --out "output/detections_$nombre.json"
    done
    python src/merge.py
"""

import argparse
import glob
import json
from pathlib import Path


def fusionar_detecciones(patron: str, out_path: Path) -> None:
    archivos = sorted(glob.glob(patron))
    if not archivos:
        raise SystemExit(f"No encontré archivos con el patrón: {patron}")

    resultados = []
    total_det = total_frames = total_vacios = 0
    modelo = umbral = None

    for path in archivos:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        video = Path(path).stem.replace("detections_", "")
        modelo, umbral = d["modelo"], d["umbral_conf"]
        for r in d["resultados"]:
            r["video"] = video
            resultados.append(r)
        total_det += d["n_detecciones_total"]
        total_frames += d["n_frames"]
        total_vacios += d["frames_sin_deteccion"]

    payload = {
        "modelo": modelo,
        "umbral_conf": umbral,
        "n_videos": len(archivos),
        "n_frames": total_frames,
        "n_detecciones_total": total_det,
        "frames_sin_deteccion": total_vacios,
        "resultados": resultados,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{len(archivos)} videos -> {out_path} ({total_frames} frames, {total_det} detecciones)")


def fusionar_metricas(patron: str, out_path: Path) -> None:
    archivos = sorted(glob.glob(patron))
    if not archivos:
        raise SystemExit(f"No encontré archivos con el patrón: {patron}")

    frames = []
    for path in archivos:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        video = Path(path).stem.replace("frames_meta_", "")
        for f in d["frames"]:
            f["video"] = video
            frames.append(f)

    payload = {"n_videos": len(archivos), "n_frames": len(frames), "frames": frames}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{len(archivos)} videos -> {out_path} ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser(description="Fusiona detections/meta de varios videos.")
    p.add_argument("--detections-glob", default="output/detections_*.json")
    p.add_argument("--meta-glob", default="output/frames_meta_*.json")
    p.add_argument("--out-detections", type=Path, default=Path("output/detections.json"))
    p.add_argument("--out-meta", type=Path, default=Path("output/frames_meta.json"))
    args = p.parse_args()

    fusionar_detecciones(args.detections_glob, args.out_detections)
    fusionar_metricas(args.meta_glob, args.out_meta)


if __name__ == "__main__":
    main()
