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
import numpy as np


def metricas_opticas(frame, umbral_quemado: int = 250) -> dict:
    """Brillo, luminancia, nitidez y clipping — en una sola conversión de color.

    Se reportan DOS medidas de brillo porque no son lo mismo y sirven para
    preguntas distintas:

    - `brillo_medio` es la media del canal V de HSV, o sea max(R,G,B). NO es
      luminancia: no pondera por respuesta espectral y un rojo saturado da
      V=255 igual que el blanco. Se usa a propósito para el clipping, porque V
      es el canal que satura primero en el sensor: un cielo quemado se detecta
      antes con V que con un promedio RGB, donde el azul lo arrastra hacia
      abajo y disimula el blowout.
    - `luminancia_media` es Rec.601 (0.299R + 0.587G + 0.114B), que es lo que
      corresponde para hablar de *exposición*. V está sistemáticamente sesgado
      hacia arriba y es ciego al color.

    `nitidez` es la varianza del laplaciano: responde a bordes, y cae cuando el
    drone gira y el motion blur los suaviza. Es un número RELATIVO — depende de
    resolución, lente y contenido de la escena — así que solo tiene sentido
    compararlo entre frames del mismo clip. analyze.py normaliza por clip
    justamente por esto.

    Los percentiles y `pct_sombras` existen porque la media es la estadística
    equivocada para una escena de alto rango dinámico: un frame puede tener
    media 110 (casi gris medio) con el cielo quemado y el primer plano en
    sombra irrecuperable. El histograma lo dice; la media lo esconde.
    """
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Rec.601
    p5, p50, p95 = (float(x) for x in np.percentile(v, [5, 50, 95]))

    return {
        "brillo_medio": round(float(v.mean()), 2),
        "luminancia_media": round(float(gris.mean()), 2),
        "nitidez": round(float(cv2.Laplacian(gris, cv2.CV_64F).var()), 2),
        "pct_quemado": round(float((v >= umbral_quemado).sum() / v.size * 100), 2),
        "pct_sombras": round(float((v <= 16).sum() / v.size * 100), 2),
        "v_p5": round(p5, 1),
        "v_p50": round(p50, 1),
        "v_p95": round(p95, 1),
        "rango_dinamico": round(p95 - p5, 1),
    }


def extraer(video_path: Path, out_dir: Path, fps_objetivo: float) -> tuple[list[dict], dict]:
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

    if fps_objetivo <= 0:
        cap.release()
        raise SystemExit(f"--fps debe ser mayor que 0 (recibido: {fps_objetivo})")
    if fps_objetivo > fps_video:
        print(
            f"Aviso: pediste {fps_objetivo} fps pero el video solo tiene "
            f"{fps_video:.2f}. Se extraerán todos los frames."
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
                    **metricas_opticas(frame),
                }
            )
            guardados += 1

        idx += 1

    cap.release()

    if guardados == 0:
        raise RuntimeError("No se extrajo ningún frame. ¿El video está vacío?")

    print(f"Guardados: {guardados} frames en {out_dir}")
    contexto = {
        "resolucion": [ancho, alto],
        "fps_video": round(fps_video, 3),
        "paso_frames": paso,
        "total_frames_video": total_frames,
    }
    return metadatos, contexto


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

    metadatos, contexto = extraer(args.video, args.out, args.fps)

    args.meta.parent.mkdir(parents=True, exist_ok=True)
    # El contexto del video se guarda para que las afirmaciones del README
    # ("1920x1080 a 24 fps") tengan respaldo en un artefacto, no en la memoria.
    payload = {
        "video": str(args.video),
        "fps_extraccion": args.fps,
        **contexto,
        "n_frames": len(metadatos),
        "frames": metadatos,
    }
    args.meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Métricas : {args.meta}")


if __name__ == "__main__":
    main()
