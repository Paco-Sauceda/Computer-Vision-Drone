"""
extract.py — video de drone → frames + métricas ópticas por frame.

Extrae 1 frame por segundo (configurable) y, de paso, mide dos cosas de cada frame:
brillo medio y nitidez. Se calculan aquí porque leer el frame ya es el paso caro;
volver a abrirlos en la Fase 2 sería trabajo duplicado.

Uso:
    python src/extract.py data/raw/vuelo01.MP4
    python src/extract.py data/raw/vuelo01.MP4 --fps 2 --out data/frames
"""

import argparse
import json
from pathlib import Path

import cv2


def brillo_medio(frame) -> float:
    """Luminancia media del frame (0-255).

    Se usa el canal V de HSV en lugar de un promedio RGB: V es max(R,G,B), que
    es lo que satura primero en el sensor. Un cielo quemado se detecta antes con
    V que con un promedio de los tres canales, donde el azul lo arrastra hacia
    abajo y disimula el clipping.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def nitidez(frame) -> float:
    """Varianza del laplaciano — proxy de nitidez.

    El laplaciano responde a cambios bruscos de intensidad, o sea bordes. Si el
    drone giraba, el motion blur suaviza los bordes y la varianza cae. Es un
    número relativo, no absoluto: solo compara frames del mismo video, misma
    resolución y mismo lente.
    """
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def porcentaje_quemado(frame, umbral: int = 250) -> float:
    """% de píxeles prácticamente clipeados en highlights.

    El brillo medio puede verse normal mientras un 15% de la escena está quemada
    sin recuperación. Ahí es donde YOLO pierde la textura del objeto.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return float((v >= umbral).sum() / v.size * 100)


def extraer(video_path: Path, out_dir: Path, fps_objetivo: float) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not fps_video or fps_video <= 0:
        cap.release()
        raise RuntimeError(
            f"El video no reporta FPS válidos (leído: {fps_video}). "
            "¿Está corrupto o es un formato raro?"
        )

    # Cada cuántos frames del video guardamos uno.
    paso = max(1, round(fps_video / fps_objetivo))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Video    : {video_path.name}")
    print(f"Resolución: {ancho}x{alto} @ {fps_video:.2f} fps, {total_frames} frames")
    print(f"Muestreo : 1 de cada {paso} frames (~{fps_objetivo} fps)")

    metadatos = []
    idx = 0
    guardados = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % paso == 0:
            segundo = idx / fps_video
            nombre = f"frame_{guardados:05d}_t{segundo:07.2f}.jpg"
            destino = out_dir / nombre
            cv2.imwrite(str(destino), frame)

            metadatos.append(
                {
                    "frame": nombre,
                    "indice_video": idx,
                    "segundo": round(segundo, 2),
                    "brillo_medio": round(brillo_medio(frame), 2),
                    "nitidez": round(nitidez(frame), 2),
                    "pct_quemado": round(porcentaje_quemado(frame), 2),
                }
            )
            guardados += 1

        idx += 1

    cap.release()

    if guardados == 0:
        raise RuntimeError("No se extrajo ningún frame. ¿El video está vacío?")

    print(f"Guardados: {guardados} frames en {out_dir}")
    return metadatos


def main():
    p = argparse.ArgumentParser(description="Extrae frames de un video de drone.")
    p.add_argument("video", type=Path, help="Ruta al video (ej. data/raw/vuelo01.MP4)")
    p.add_argument("--fps", type=float, default=1.0, help="Frames por segundo a extraer (default: 1)")
    p.add_argument("--out", type=Path, default=Path("data/frames"), help="Carpeta destino de los frames")
    p.add_argument(
        "--meta",
        type=Path,
        default=Path("output/frames_meta.json"),
        help="JSON con las métricas por frame",
    )
    args = p.parse_args()

    if not args.video.exists():
        raise SystemExit(f"No existe el archivo: {args.video}")

    metadatos = extraer(args.video, args.out, args.fps)

    args.meta.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(args.video),
        "fps_extraccion": args.fps,
        "n_frames": len(metadatos),
        "frames": metadatos,
    }
    args.meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Métricas : {args.meta}")


if __name__ == "__main__":
    main()
