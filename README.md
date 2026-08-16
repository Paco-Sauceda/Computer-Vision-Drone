# drone-vision

> Aerial video → object detection → **an analysis of why detection fails under real capture conditions.**

<!-- TODO: result image or GIF here (output/annotated/) -->

---

## Why I built this

I'm a Computer Science student and a photographer. I own the drone, two cameras and four lenses used to shoot the footage in this project.

Any CS student can call YOLO. Far fewer can explain *why* the model breaks under backlight, motion blur, or a nadir camera angle — because that requires understanding exposure, sensor behaviour and optics, not just an API.

That analysis is the point of this repo. The pipeline is just what makes the analysis possible.

---

## Findings

<!-- TODO Phase 2 — three concrete findings, each with a chart.
     Example of the shape they should take:
     "Mean confidence drops 40% once frame brightness exceeds 200 —
      highlight clipping on bright surfaces destroys the texture YOLO relies on." -->

*Pending — Phase 2.*

---

## How it works

```
video ──► extract.py ──► frames + optical metrics ──► detect.py ──► detections.json
                         (brightness, sharpness,                    (class, confidence,
                          highlight clipping)                        bbox, per frame)
```

`extract.py` measures three things per frame while it decodes:

| Metric | How | Why it matters |
|---|---|---|
| **Mean brightness** | HSV *V* channel mean | *V* = max(R,G,B), so it catches highlight clipping before an RGB average does — blue sky drags an RGB mean down and hides the blowout |
| **Sharpness** | Variance of the Laplacian | Drops when the drone yaws; motion blur softens the edges detectors depend on |
| **% clipped** | Share of pixels with V ≥ 250 | Mean brightness can look fine while 15% of the scene is unrecoverable |

These are computed during extraction, not later — decoding the frame is the expensive step, and Phase 2 shouldn't pay it twice.

---

## Run it

```bash
git clone <repo-url>
cd drone-vision
pip install -r requirements.txt

# put a drone video in data/raw/, then:
python src/extract.py data/raw/your_flight.MP4
python src/detect.py --annotate
```

Output lands in `output/detections.json`, already joined with the per-frame optical metrics.

Runs on CPU with `yolov8n.pt`. Weights download automatically on first run.

**Options**

```bash
python src/extract.py data/raw/flight.MP4 --fps 2      # sample 2 frames/second
python src/detect.py --conf 0.35                        # raise confidence threshold
python src/detect.py --modelo yolov8s.pt                # bigger model, slower
```

---

## Limitations

- **Pretrained on COCO.** The classes are ground-level categories — `person`, `car`, `truck`. COCO contains almost no nadir aerial imagery, so low confidence on overhead shots is a *domain gap*, not a bug. Quantifying that gap is part of the point.
- **No tracking.** Each frame is scored independently. An object detected in frame 4 and missed in frame 5 counts as two separate events.
- **Sharpness is relative.** Laplacian variance depends on resolution, lens and scene content. Only compare frames from the same clip.
- **No ground truth.** There are no manual annotations, so this measures *confidence*, not accuracy. A confident wrong detection looks the same as a confident right one here.
- **No altitude from telemetry yet.** Altitude is annotated by hand.

---

## AI assistance

<!-- TODO — fill in honestly as you go. What was generated, what you corrected,
     what bug you caught. This section is not filler: it demonstrates you can
     audit generated code, which is what employers are actually screening for. -->

*Pending.*
