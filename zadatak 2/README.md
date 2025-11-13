# Buzzy Beetle Counting (Kolokvijum)

This is my solution for a small computer vision task: count how many blue "Buzzy Beetles" pass through the middle of each video and report the error against the provided ground-truth counts.

## What was the task

- Input: 10 short videos in `data/` and a CSV file `buzzy_beetle_count.csv` with the true number of beetles per video.
- Goal: Predict the beetle count for each video and compute Mean Absolute Error (MAE) over all videos.
- Metric: MAE (lower is better).

## Result

- Final MAE: **0.2**
- Both implementations print the MAE to one decimal place.
- Alternate notebook (`mikuta.ipynb`) achieved **~0.3 MAE** during exploration.

## What I did to achieve that

I iterated on two simple classical CV pipelines (no ML training), experimenting in the notebooks and then packaging the logic in Python scripts.

### 1) `kolokvijum.py` (final run I used)

Pipeline highlights:

- Region of interest (ROI): I cropped the video to a relevant area to reduce noise: `(x=400, y=250, w=500, h=320)`.
- Background subtraction: MOG2 to isolate moving objects, followed by small denoising (median blur).
- Contour filtering: I kept contours in expected area ranges. For large blobs, I split them into a grid of subregions to avoid undercounting when multiple beetles touched.
- Color check in HSV: I validated that the candidate region is within a dark/blue HSV range (tight thresholds) to avoid false positives.
- Crossing rule: I counted a beetle only when its center passed near the vertical center line of the ROI. I used a short skip window and a set of IDs to avoid double counting on adjacent frames.
- Evaluation: For each `video_1.mp4` … `video_10.mp4`, I computed the predicted count and then MAE against the CSV.

Why this worked: the beetles are blue and moving; combining motion (BG subtractor), geometry (contour area), and color (HSV) gave robust detections. The center-line crossing rule stabilized the final count.

### 2) `klk.py` (earlier attempt)

- Cropped a centered square portion of the frame.
- Used HSV masking for dark blue, found contours within an area range.
- Tracked approximate centers to avoid recounting within a distance threshold; counted when passing a chosen line.

This was simpler but a bit more sensitive to illumination and blob merging, so I used `kolokvijum.py` for the final score.

### 3) `mikuta.ipynb` (notebook exploration; MAE ~0.3)

An exploratory notebook that implements a simpler HSV-based pipeline with visualization:

- Cropping: takes a centered square region with a downward offset (roughly focuses on the lane): uses `start_y = (h - size) // 2 + 200` and `size - 100`.
- Color segmentation: converts to HSV and thresholds a dark-blue range `lower_blue = [85, 80, 45]`, `upper_blue = [140, 255, 255]` to get a binary mask.
- Contours: finds external contours and keeps areas between 1000 and 2000.
- De-duplication: maintains a list of previously seen centers and considers a detection new only if it’s farther than a `distance_threshold = 40` pixels.
- Crossing rule: counts when the detected center crosses a vertical `track_line = 450`.
- Visualization: draws bounding boxes and the vertical line; optionally shows frames while counting; prints a per-video table and the overall MAE.

This approach worked reasonably well and was faster to iterate on due to inline visualization, but it was more prone to missed counts when blobs merged or illumination changed, hence the slightly higher MAE.

## How to run (Windows PowerShell)

Prerequisites:

- Python 3.9+ (I used CPython),
- Packages: `opencv-python`, `numpy`, `pandas`.

Install dependencies:

```powershell
python -m pip install --upgrade pip ; pip install opencv-python numpy pandas
```

Run the script (prints MAE to stdout):

```powershell
# From the repository root
python kolokvijum.py data
```

You can also try the alternate pipeline:

```powershell
python klk.py data
```

## Repository layout

- `kolokvijum.py` — Final counting pipeline (BG subtractor + HSV + center-line crossing).
- `klk.py` — Alternate pipeline (HSV mask + simple tracking).
- `Kolokvijum.ipynb`, `mikuta.ipynb`, `primervezbe.ipynb`, `jedanvideo.ipynb` — Notebooks I used for exploration and visualization (`mikuta.ipynb` contains the HSV + contour + simple tracking approach that scored ~0.3 MAE).
- `data/` — Contains `video_1.mp4` … `video_10.mp4` and `buzzy_beetle_count.csv` with ground-truth counts.
