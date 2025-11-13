# Computer Vision Mini-Projects — Zadatak 1 & Zadatak 2

I combined both of my small CV tasks into a single README so it’s easy to see the approaches, how to run them, and the results. Everything here is written in first person, as if I’m documenting my own work.

- Zadatak 1: Duck counting on images (two scripts: `resenje.py`, `mikutapi.py`).
- Zadatak 2: Buzzy Beetle counting in videos (mainly `kolokvijum.py`, plus `klk.py` and notebooks).

Use the table of contents below to jump to each task.

- [Zadatak 1 — Duck Counting (images)](#zadatak-1--duck-counting-images)
- [Zadatak 2 — Buzzy Beetle Counting (videos)](#zadatak-2--buzzy-beetle-counting-videos)

---

## Zadatak 1 — Duck Counting (images)

This part contains two independent solutions for the same task: counting rubber ducks on images and reporting the overall MAE (Mean Absolute Error) against ground-truth labels from a CSV file.

- Main solution: `resenje.py`
- Alternative solution: `mikutapi.py`

Both scripts take the dataset folder path as a single argument and print one number to stdout — the MAE over the full set.

### Dataset layout

Expected folder structure:

```
data/
  duck_count.csv          # columns: picture, ducks
  picture_1.jpg
  picture_2.jpg
  ...
  picture_10.jpg
```

CSV (`duck_count.csv`) must contain at least:

- `picture`: image filename (e.g. `picture_1.jpg`)
- `ducks`: ground-truth count for that image (integer)

Important: the filenames in the CSV have to match the actual files under `data/`.

### What I used (tech stack)

- Python 3.10+
- OpenCV (`opencv-python`) for image processing
- NumPy for array ops
- pandas for reading the CSV and computing MAE

### Setup (Windows PowerShell)

I recommend a virtual environment:

```powershell
python -m venv .venv; .\\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip
python -m pip install opencv-python numpy pandas
```

### How to run

Each script expects one argument: the dataset folder (e.g. `data`). It prints the MAE to stdout.

```powershell
# Main solution
python resenje.py data

# Alternative solution
python mikutapi.py data
```

### Results (MAE)

On my dataset and settings, I measured:

- `resenje.py` (main): MAE = 0.2
- `mikutapi.py` (alternative): MAE = 0.5

These values depend on the dataset and parameters (thresholds, kernels, ROI/mask).

### How I solved it — design and rules

I kept the approach deterministic and classical (no training), relying on simple but well-structured image processing steps. Key rules and decisions:

- Fixed preprocessing pipeline per solution (no randomness; same inputs → same outputs).
- Threshold-based binarization tuned to this dataset’s lighting/background.
- Morphological operations to reduce noise and close small gaps.
- Contour filling and area-based filtering as the main heuristic.
- Consistent naming and a small, self-contained CLI: one argument (dataset folder), one numeric output (MAE).

#### `resenje.py` — edges + filled contours over a fixed ROI

1. Crop a fixed ROI: `img[250:800, 200:800]`.
2. Grayscale → `THRESH_BINARY_INV` (threshold ~80).
3. Morphological close (ellipse 5×5, 2 iterations).
4. Canny edges → dilate/erode → close.
5. Fill external contours; filter by area (px²): 750–2100, 2200–5000, 6700–7000, 7500–10000; also avoid border-touching blobs.
6. Light dilation (4×4 kernel, 3 iterations), re-detect contours and count.
7. Loop `picture_1.jpg` … `picture_10.jpg`, compute MAE vs CSV.

Why it works: the ROI avoids distractions; the edge→morph→fill sequence yields stable blobs with consistent areas.

#### `mikutapi.py` — central crop + circular mask with a robust fallback

1. Central crop (20%..80% width/height), circular mask to ignore corners.
2. Gaussian blur (7×7) → grayscale → `THRESH_BINARY_INV` (~85).
3. Morphological close then open (5×5).
4. Fill contours and filter by area: main `min_area..max_area` (typ. 400–5000) plus ~5980–6300 and ~7700–9000 px².
5. Fallback: shrink mask radius to 70%, stronger dilation (10×10, 2 iters), retry.
6. Loop over 10 images, compute MAE vs CSV.

Why it works: the circular mask + central crop reduces edge false positives; the fallback helps when the initial pass is too conservative.

---

## Zadatak 2 — Buzzy Beetle Counting (videos)

This is my solution for a small computer vision task: count how many blue "Buzzy Beetles" pass through the middle of each video and report the error against the provided ground-truth counts.

### What was the task

- Input: 10 short videos in `data/` and a CSV file `buzzy_beetle_count.csv` with the true number of beetles per video.
- Goal: Predict the beetle count for each video and compute Mean Absolute Error (MAE) over all videos.
- Metric: MAE (lower is better).

### Result

- Final MAE: **0.2**
- Both implementations print the MAE to one decimal place.
- Alternate notebook (`mikuta.ipynb`) achieved **~0.3 MAE** during exploration.

### What I did to achieve that

I iterated on two simple classical CV pipelines (no ML training), experimenting in the notebooks and then packaging the logic in Python scripts.

#### 1) `kolokvijum.py` (final run I used)

Pipeline highlights:

- Region of interest (ROI): cropped the video to a relevant area: `(x=400, y=250, w=500, h=320)`.
- Background subtraction: MOG2 to isolate moving objects, followed by small denoising (median blur).
- Contour filtering: kept contours in expected area ranges. For large blobs, split into a grid of subregions to avoid undercounting when multiple beetles touched.
- Color check in HSV: validated candidates in a dark/blue HSV range (tight thresholds) to avoid false positives.
- Crossing rule: counted a beetle only when its center passed near the vertical center line of the ROI; used a short skip window and a set of IDs to avoid double counting.
- Evaluation: for each `video_1.mp4` … `video_10.mp4`, computed the predicted count and then MAE against the CSV.

Why this worked: the beetles are blue and moving; combining motion (BG subtractor), geometry (contour area), and color (HSV) gave robust detections. The center-line crossing rule stabilized the final count.

#### 2) `klk.py` (earlier attempt)

- Cropped a centered square portion of the frame.
- Used HSV masking for dark blue, found contours within an area range.
- Tracked approximate centers to avoid recounting within a distance threshold; counted when passing a chosen line.

This was simpler but more sensitive to illumination and blob merging, so I used `kolokvijum.py` for the final score.

#### 3) `mikuta.ipynb` (notebook exploration; MAE ~0.3)

An exploratory notebook that implements a simpler HSV-based pipeline with visualization:

- Cropping: takes a centered square region with a downward offset (`start_y = (h - size) // 2 + 200`, `size - 100`).
- Color segmentation: HSV thresholding with `lower_blue = [85, 80, 45]`, `upper_blue = [140, 255, 255]` to get a binary mask.
- Contours: finds external contours and keeps areas between 1000 and 2000.
- De-duplication: keeps previously seen centers; a detection is new only if farther than `distance_threshold = 40` pixels.
- Crossing rule: counts when the detected center crosses `track_line = 450`.
- Visualization: draws bounding boxes and the vertical line; optionally shows frames; prints a per-video table and the overall MAE.

This approach worked reasonably well and was fast to iterate on, but it was more prone to missed counts when blobs merged or illumination changed, hence the slightly higher MAE.

### How to run (Windows PowerShell)

Prerequisites:

- Python 3.9+ (I used CPython)
- Packages: `opencv-python`, `numpy`, `pandas`

Install dependencies:

```powershell
python -m pip install --upgrade pip ; pip install opencv-python numpy pandas
```

Run the script (prints MAE to stdout):

```powershell
# From the repository root
python kolokvijum.py data
```

Try the alternate pipeline:

```powershell
python klk.py data
```

### Repository layout (key files)

- Duck counting (images): `resenje.py`, `mikutapi.py`, and `data/` with `duck_count.csv` and `picture_*.jpg`.
- Buzzy beetle counting (videos): `kolokvijum.py`, `klk.py`, notebooks (`mikuta.ipynb`, etc.), and `data/` with `video_*.mp4` and `buzzy_beetle_count.csv`.
