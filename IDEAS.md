# IDEAS — parking lot

Todo lo que se le ocurra a mitad de una fase y no toque implementar todavía va aquí.
Regla: si está en este archivo, **no** se implementa hasta terminar la fase actual.

## Siguientes, ordenadas por valor por hora

1. **`--imgsz 1280` sobre un solo clip** (el 0006, 155 frames, es el que más contenido tiene).
   Mismos frames, misma escena, misma luz, una variable. Separa "domain gap de COCO"
   de "resolución de inferencia", que hoy están confundidos. ~20 min de código, ~40 min
   de CPU desatendida. La bandera ya está expuesta en `detect.py`.
2. **Barrido de blur sintético.** 30 frames nítidos del clip 0009 + `cv2.GaussianBlur`
   con σ creciente + YOLO en cada nivel. Convierte el hallazgo 2 (observacional y roto)
   en un experimento controlado. Predicción a registrar *antes* de correrlo: por debajo
   de σ≈3 no pasa casi nada, porque el downsample a 640 ya destruyó esa información.
3. **Leer telemetría XMP de DJI** — pitch de gimbal, altitud relativa, GPS. Convierte
   "cenital a ojo" en una medición y habilita GSD como variable. Es la feature no
   construida de mayor valor del repo.
4. **Anotar a mano ~100 frames estratificados** (30 buenos, 30 oscuros, 30 con blur,
   10 cenitales) para poder decir *recall* y *precision* de verdad en un subconjunto.
   Medio día. Cierra el flanco de "¿detection rate contra qué ground truth?".
5. **Re-extraer con las métricas de histograma** ahora que `extract.py` guarda
   `pct_sombras`, `v_p5/p50/p95` y `rango_dinamico`. Permite sostener la afirmación
   que hoy no se puede: no es subexposición, es rango dinámico.
6. **GIF de cabecera** desde `output/annotated/` con `ffmpeg`. Impacto visual alto,
   valor técnico bajo. Hacerlo el día antes de mandar solicitudes.

## Sin clasificar

- Comparar yolov8n vs yolov8s para cuantificar el trade-off velocidad/confianza
- Inferencia por lotes en `detect.py` (`stream=True` / lista de rutas) — solo vale la
  pena si se van a reprocesar los 603 frames varias veces
- Pinnear `ultralytics==` en `requirements.txt` (la versión ya se guarda en el JSON)
- Comparar contra un detector open-vocabulary (Grounding DINO, Moondream) bajo las
  mismas condiciones: la hipótesis es que falla distinto — alucina categoría en vez
  de no detectar
