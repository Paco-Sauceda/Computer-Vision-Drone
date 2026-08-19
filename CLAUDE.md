# drone-vision — contexto del proyecto

Claude Code lee este archivo automáticamente. Con esto arranca sabiendo qué construir y qué no.

---

## Qué es esto y para qué

Pipeline que analiza video aéreo de drone con visión por computadora y produce un reporte de qué se detectó y **por qué la detección falla** en ciertas condiciones.

**El objetivo real no es el código. Es tener algo que enseñar en octubre** cuando aplique a puestos de Werkstudent en Alemania — congatec, startups del ITC1, y el Technology Campus Plattling de THD, que investiga sistemas autónomos.

**Contexto del autor:** estudiante de Ciencias Computacionales y fotógrafo. Tiene drone propio, dos cámaras y cuatro lentes. Entiende óptica, exposición y comportamiento de sensores a nivel físico — no solo a nivel de API.

**Esa es la tesis del proyecto:** cualquier estudiante de CS puede llamar a YOLO. Muy pocos pueden explicar *por qué* el modelo falla con contraluz, movimiento o ángulo cenital. Ese análisis es el diferenciador, no el pipeline.

---

## ⚠️ Restricciones — respétalas

**Tiempo:** 4 semanas, y no es dedicación completa.

**Por lo tanto:**

- **Modesto y terminado gana a ambicioso e incompleto.** Siempre.
- Nada de entrenar modelos. Solo modelos preentrenados.
- Nada de interfaz web, ni base de datos, ni Docker, ni tests exhaustivos.
- Si una decisión añade más de un día de trabajo, la respuesta por defecto es no.
- Prioriza que funcione de punta a punta sobre que esté bien arquitecturado.

**Si el autor pide algo que amplíe el alcance, recuérdale esta sección antes de implementarlo.**

---

## Stack

```
Python 3.11+
ultralytics      # YOLOv8 preentrenado
opencv-python    # lectura de video, frames
pandas           # análisis de resultados
matplotlib       # gráficas del análisis
```

Sin GPU. Todo corre en CPU con modelos pequeños (`yolov8n.pt`). Si es lento, se procesa 1 de cada N frames.

---

## Estructura

```
drone-vision/
├── CLAUDE.md              ← este archivo
├── README.md              ← el entregable que leerán los reclutadores
├── requirements.txt
├── data/
│   ├── raw/               ← videos originales (gitignore)
│   └── frames/            ← frames extraídos (gitignore)
├── src/
│   ├── extract.py         ← video → frames
│   ├── detect.py          ← frames → detecciones JSON
│   ├── analyze.py         ← detecciones → estadísticas
│   └── report.py          ← estadísticas → reporte markdown
├── output/
│   ├── detections.json
│   ├── stats.csv
│   └── charts/
└── notebooks/
    └── analisis.ipynb     ← exploración y gráficas
```

---

## Fases

### Fase 1 — que funcione de punta a punta (fin de semana 1)

**Meta:** meter un video, obtener un JSON con detecciones.

- [x] `extract.py` — extraer 1 frame por segundo de un video
- [x] `detect.py` — correr YOLOv8n sobre cada frame, guardar clase, confianza y bounding box
- [x] Salida: `output/detections.json`
- [x] Probarlo con **un solo video corto real de drone** (30–60 segundos) — terminó siendo 10 videos reales, 27s–155s cada uno

**Criterio de terminado:** un comando produce el JSON sin errores.

Aquí termina lo obligatorio. Si el tiempo se acaba, esto ya es publicable.

### Fase 2 — el análisis que diferencia (fin de semana 2)

**Meta:** responder *por qué* falla la detección.

- [x] `analyze.py` — agregar por clase: cuántos objetos, confianza media, distribución
- [x] Cruzar confianza contra condiciones de captura:
  - [ ] **Altitud** — no disponible en telemetría de este drone, no se anotó a mano
  - [ ] **Hora del día / ángulo del sol** — pendiente, sí se ve el efecto indirecto en el hallazgo 3 (golden hour → falso positivo "kite")
  - [x] **Brillo medio del frame** (ya se calcula en `extract.py`)
  - [x] **Nitidez** (varianza del laplaciano — ya se calcula en `extract.py`)
- [x] Gráficas: tasa de detección vs brillo, vs nitidez (más confianza vs brillo/nitidez/% quemado)
- [x] Escribir **tres hallazgos concretos** en el README

**Ejemplos del tipo de hallazgo que se busca:**
> "La confianza cae 40% cuando el brillo medio del frame supera 200 — sobreexposición en superficies claras."
> "Por debajo de X de nitidez, la detección de peatones desaparece: el motion blur del drone en giro destruye los bordes."

**Esto es el corazón del proyecto.** Un estudiante de CS produce el pipeline. Un fotógrafo que programa produce este análisis.

### Fase 3 — opcional, solo si sobra tiempo

- [x] `report.py` — generar un reporte markdown legible desde las estadísticas
- [ ] Integrar un LLM para redactar el resumen en lenguaje natural — descartado, agrega costo de API sin aportar al diferenciador del proyecto
- [ ] Automatizar: carpeta vigilada que dispara el pipeline al detectar un video nuevo — descartado, fuera de alcance para una pieza de portafolio

**Si la Fase 2 no está terminada, no empieces la Fase 3.**

---

## El README es el entregable real

Los reclutadores no van a leer tu código. Van a leer el README y ver las gráficas.

Debe tener, en este orden:

1. **Una frase** de qué hace, con una imagen o GIF de resultado
2. **Por qué lo hice** — el ángulo de fotógrafo + programador, explícito
3. **Los tres hallazgos**, con gráficas
4. **Cómo correrlo** — instalación y comando, que funcione copiando y pegando
5. **Limitaciones** — qué no hace y por qué. Esto demuestra criterio
6. **Si usaste asistentes de IA**: dónde, qué corregiste, qué error detectaste

Ese último punto no es relleno. En el mercado alemán se busca gente que sepa **auditar** código generado por IA, no solo generarlo. Documentarlo demuestra criterio técnico y es raro de ver en portafolios de estudiantes.

Escribe el README **en inglés**.

---

## Cómo trabajar con Claude Code en esto

- Empieza cada sesión diciendo en qué fase estás
- Pide **una función a la vez**, no el proyecto completo
- Después de cada pieza: córrela con datos reales antes de seguir
- Cuando algo falle, comparte el error completo, no un resumen
- Al terminar cada fase: commit con mensaje descriptivo

**Regla contra el scope creep:** si en medio de la Fase 1 surge una idea genial para la Fase 3, anótala en un `IDEAS.md` y sigue. No la implementes.

---

## Definición de éxito

Un repo público con:

- ✅ Código que corre de punta a punta con un comando
- ✅ README en inglés con imagen de resultado y tres hallazgos
- ✅ Al menos una gráfica del análisis
- ✅ Historial de commits que se vea como trabajo real, no un único volcado

**No es requisito:** que sea rápido, que soporte muchos formatos, que tenga tests, que escale.

Un proyecto pequeño y terminado, con un análisis que solo tú podías hacer, vale más que uno grande a medias.
