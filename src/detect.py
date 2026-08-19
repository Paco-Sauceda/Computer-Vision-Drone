"""
detect.py — frames → detecciones JSON.

Corre YOLOv8n (preentrenado en COCO) sobre cada frame extraído y guarda clase,
confianza y bounding box. No entrena nada. CPU.

El JSON de salida ya viene unido con las métricas ópticas de extract.py, para que
la Fase 2 sea un solo pandas.read_json y no un join a mano.

Uso:
    python src/detect.py
    python src/detect.py --frames data/frames --conf 0.35 --annotate
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

EXTENSIONES = {".jpg", ".jpeg", ".png"}


def _version_ultralytics() -> str:
    """Se guarda en el JSON: sin esto, "corre esto y obtén 387 detecciones" no
    es una promesa que se pueda cumplir dentro de seis meses."""
    try:
        from importlib.metadata import version

        return version("ultralytics")
    except Exception:
        return "desconocida"


def cargar_metricas(meta_path: Path) -> dict[str, dict]:
    """Devuelve {nombre_frame: métricas} desde output/frames_meta.json, si existe."""
    if not meta_path.exists():
        print(f"Aviso: no encontré {meta_path}. Sigo sin métricas ópticas.")
        return {}
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return {f["frame"]: f for f in data.get("frames", [])}


def detectar(
    frames_dir: Path,
    modelo_path: str,
    conf: float,
    imgsz: int,
    metricas: dict[str, dict],
    annotate_dir: Path | None,
) -> list[dict]:
    frames = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() in EXTENSIONES)
    if not frames:
        raise RuntimeError(f"No hay frames en {frames_dir}. ¿Corriste extract.py primero?")

    print(f"Modelo   : {modelo_path}")
    print(f"Frames   : {len(frames)} en {frames_dir}")
    print(f"Confianza: umbral {conf}")
    print(f"imgsz    : {imgsz} (un frame 1920x1080 se reescala a ~{imgsz}x{round(imgsz*1080/1920)})")

    modelo = YOLO(modelo_path)
    if annotate_dir:
        annotate_dir.mkdir(parents=True, exist_ok=True)

    resultados = []

    for i, frame_path in enumerate(frames, start=1):
        # verbose=False porque si no, ultralytics imprime una línea por frame.
        pred = modelo.predict(source=str(frame_path), conf=conf, imgsz=imgsz, verbose=False)[0]

        detecciones = []
        for box in pred.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0])
            detecciones.append(
                {
                    "clase": modelo.names[cls_id],
                    "clase_id": cls_id,
                    "confianza": round(float(box.conf[0]), 4),
                    "bbox": [x1, y1, x2, y2],
                    "area_px": round((x2 - x1) * (y2 - y1), 1),
                }
            )

        registro = {
            "frame": frame_path.name,
            "n_detecciones": len(detecciones),
            "confianza_media": (
                round(sum(d["confianza"] for d in detecciones) / len(detecciones), 4)
                if detecciones
                else None
            ),
            "detecciones": detecciones,
        }

        # Pegar brillo / nitidez / % quemado si extract.py los generó.
        m = metricas.get(frame_path.name)
        if m:
            registro["segundo"] = m["segundo"]
            registro["brillo_medio"] = m["brillo_medio"]
            registro["nitidez"] = m["nitidez"]
            registro["pct_quemado"] = m["pct_quemado"]

        resultados.append(registro)

        if annotate_dir:
            pred.save(filename=str(annotate_dir / frame_path.name))

        if i % 10 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)} frames procesados")

    return resultados


def main():
    p = argparse.ArgumentParser(description="Detecta objetos en los frames extraídos.")
    p.add_argument("--frames", type=Path, default=Path("data/frames"), help="Carpeta de frames")
    p.add_argument("--modelo", default="yolov8n.pt", help="Pesos YOLO preentrenados")
    p.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza (default: 0.25)")
    p.add_argument(
        "--meta",
        type=Path,
        default=Path("output/frames_meta.json"),
        help="JSON de métricas de extract.py",
    )
    p.add_argument(
        "--out", type=Path, default=Path("output/detections.json"), help="JSON de salida"
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help=(
            "Resolución de inferencia (default: 640, el default de ultralytics). "
            "Un frame 1920x1080 entra al modelo reescalado a ~640x360: un noveno "
            "del área. Subirlo a 1280 es el experimento controlado más barato "
            "que tiene este repo."
        ),
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Guardar frames con bounding boxes (ver --annotate-dir)",
    )
    p.add_argument(
        "--annotate-dir",
        type=Path,
        default=None,
        help="Carpeta destino de los frames anotados (default: output/annotated/)",
    )
    args = p.parse_args()

    if not args.frames.exists():
        raise SystemExit(f"No existe la carpeta de frames: {args.frames}")

    metricas = cargar_metricas(args.meta)
    annotate_dir = args.annotate_dir or (Path("output/annotated") if args.annotate else None)
    resultados = detectar(args.frames, args.modelo, args.conf, args.imgsz, metricas, annotate_dir)

    total = sum(r["n_detecciones"] for r in resultados)
    vacios = sum(1 for r in resultados if r["n_detecciones"] == 0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "modelo": args.modelo,
        "umbral_conf": args.conf,
        "imgsz": args.imgsz,
        "ultralytics_version": _version_ultralytics(),
        "n_frames": len(resultados),
        "n_detecciones_total": total,
        "frames_sin_deteccion": vacios,
        "resultados": resultados,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nDetecciones: {total} en {len(resultados)} frames")
    print(f"Frames sin ninguna detección: {vacios}")
    print(f"Salida: {args.out}")
    if annotate_dir:
        print(f"Frames anotados: {annotate_dir}")


if __name__ == "__main__":
    main()
