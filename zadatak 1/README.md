# Duck Counting — My Approach (Computer Vision)

This repository contains two independent solutions for the same task: counting rubber ducks on images and reporting the overall MAE (Mean Absolute Error) against ground-truth labels from a CSV file.

- Main solution: `resenje.py`
- Alternative solution: `mikutapi.py`

Both scripts take the dataset folder path as a single argument and print one number to stdout — the MAE over the full set.

I wrote this README in first person to explain how I approached the problem, which technologies I used, the rules I followed, and how to run everything.

## Dataset layout

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

## What I used (tech stack)

- Python 3.10+
- OpenCV (`opencv-python`) for image processing
- NumPy for array ops
- pandas for reading the CSV and computing MAE

## Setup (Windows PowerShell)

I recommend a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install --upgrade pip
pip install opencv-python numpy pandas
```

## How to run

Each script expects one argument: the dataset folder (e.g. `data`). It prints the MAE to stdout.

```powershell
# Main solution
python resenje.py data

# Alternative solution
python mikutapi.py data
```

## Results (MAE)

On my dataset and settings described below, I measured the following mean absolute errors:

- `resenje.py` (main solution): MAE = 0.2
- `mikutapi.py` (alternative): MAE = 0.5

## How I solved it — design and rules

I decided to keep the approach deterministic and classical (no training), relying on simple but well-structured image processing steps. Key rules and decisions I followed:

- Fixed, explicit preprocessing pipeline per solution (no randomness; same inputs → same outputs).
- Threshold-based binarization tuned to this dataset’s lighting/background.
- Morphological operations to reduce noise and close small gaps before counting objects.
- Contour filling and area-based filtering as the main heuristic for separating ducks from noise.
- Consistent naming and a small, self-contained script interface: one CLI argument (dataset folder), one numeric output (MAE).

Below is what each script does and why.

### `resenje.py` — edges + filled contours over a fixed ROI

My focus here was on edge consolidation and contour filling inside a hand-picked region of interest (ROI):

1. Crop a fixed ROI: `img[250:800, 200:800]` — this removes borders and background clutter I observed in the dataset.
2. Grayscale → binary with `THRESH_BINARY_INV` (threshold ~80) to highlight ducks against the background.
3. Morphological close (ellipse 5×5, 2 iterations) to connect fragmented parts.
4. Canny edges, then dilate/erode and one more close to seal open edges.
5. Fill external contours; filter candidates by area using several hand-tuned ranges that matched duck sizes in this set:
   - 750–2100, 2200–5000, 6700–7000, 7500–10000 (px²)
     I also guard against contours touching the very border of the ROI.
6. Light dilation (small 4×4 kernel, 3 iterations), then re-detect external contours and count them.
7. Repeat for `picture_1.jpg` … `picture_10.jpg`, compare to CSV, and print MAE.

Why this works here: the ROI avoids most distractions, while the edge→morphology→fill sequence gives stable blobs whose areas are consistent enough to filter reliably.

### `mikutapi.py` — central crop + circular mask with a robust fallback

This variant assumes ducks are mostly central and suppresses corner noise via a circular mask:

1. Central crop (20%..80% of width/height), then apply a circular mask to ignore corners.
2. Gaussian blur (7×7) → grayscale → `THRESH_BINARY_INV` with a threshold around 85.
3. Morphological close then open (5×5) to reduce noise and separate blobs.
4. Fill contours and filter by area. I use a main range and a couple of larger ranges observed in the data:
   - main: `min_area..max_area` (typically 400–5000)
   - extra: ~5980–6300 and ~7700–9000 (px²)
5. Fallback if nothing is found: shrink the mask radius to 70%, apply stronger dilation (10×10 kernel, 2 iters), and try again. This recovers cases where ducks are small or faint.
6. Loop over all 10 images, compute MAE vs CSV, and print it.

Why this works here: the circular mask plus central crop reduces false positives near edges, and the fallback helps when the initial thresholding under-segments the ducks.
