# EDA Report - MOT Dataset

## Scope
- Detection statistics: confidence, detections-per-frame, bbox size distributions.
- Sequence consistency checks: frame index continuity and value ranges.
- Visual sanity checks: detections overlaid with GT on sample frames.

## Global Summary
- Sequences analyzed: 4
- Total detections: 45004
- Total GT rows (if available): 143879
- Total invalid bboxes (w<=0 or h<=0): 0

## Sequence Summaries
### MOT_02
- Detections: 28406 | GT rows: 108005 | GT eval class=1: 47557
- Confidence: min=0.050, q05=0.989, median=1.000, q95=1.000, max=1.000, mean=0.980
- Detections/frame: q05=21.00, median=28.00, q95=33.00, mean=27.05
- BBox area: q05=6677.60, median=10410.28, q95=23791.90; mean width=66.41, mean height=180.20
- Frame checks (DET): min=1, max=1050, missing=0, starts_at_1=True
- Frame checks (GT): min=1, max=1050, missing=0, starts_at_1=True

### MOT_03
- Detections: 3848 | GT rows: 8013 | GT eval class=1: 6917
- Confidence: min=0.050, q05=0.222, median=1.000, q95=1.000, max=1.000, mean=0.927
- Detections/frame: q05=2.00, median=4.00, q95=8.00, mean=4.60
- BBox area: q05=1188.54, median=21902.40, q95=124656.00; mean width=109.99, mean height=271.85
- Frame checks (DET): min=1, max=837, missing=0, starts_at_1=True
- Frame checks (GT): min=1, max=837, missing=0, starts_at_1=True

### MOT_04
- Detections: 3049 | GT rows: 10411 | GT eval class=1: 5325
- Confidence: min=0.050, q05=0.829, median=1.000, q95=1.000, max=1.000, mean=0.966
- Detections/frame: q05=4.00, median=6.00, q95=8.00, mean=5.81
- BBox area: q05=11770.38, median=38756.30, q95=288497.92; mean width=157.75, mean height=391.97
- Frame checks (DET): min=1, max=525, missing=0, starts_at_1=True
- Frame checks (GT): min=1, max=525, missing=0, starts_at_1=True

### MOT_05
- Detections: 9701 | GT rows: 17450 | GT eval class=1: 12839
- Confidence: min=0.050, q05=0.227, median=1.000, q95=1.000, max=1.000, mean=0.904
- Detections/frame: q05=11.00, median=15.00, q95=19.00, mean=14.83
- BBox area: q05=1284.40, median=5152.30, q95=109180.76; mean width=58.62, mean height=157.76
- Frame checks (DET): min=1, max=654, missing=0, starts_at_1=True
- Frame checks (GT): min=1, max=654, missing=0, starts_at_1=True

## Observations That May Hurt Tracking
- MOT_02: dense frames likely (det/frame q95=33.0)

## Generated Visualizations
- `MOT_02_preview.png`
- `MOT_03_preview.png`
- `MOT_04_preview.png`

## Next Recommendations
- Start with a robust confidence threshold search (e.g. 0.2 to 0.6).
- Tune max_age jointly with IoU threshold to control FN vs IDSW trade-off.
- Inspect sequences with dense frames first if ID switches are high.