import sys
import cv2
import numpy as np
import pandas as pd

def count_ducks_with_filled_contours(image_path):
    img = cv2.imread(image_path)
    crop_img = img[250:800, 200:800]

    grayscale_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    _, binary_img = cv2.threshold(grayscale_img, 80, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(closed_img, threshold1=50, threshold2=150)

    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    dilated_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    filled_img = np.zeros_like(grayscale_img)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 750 < area < 2100 or 2200 < area < 5000 or 6700 < area < 7000 or 7500 < area < 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            if x > 0 and y > 0 and x + w < filled_img.shape[1] and y + h < filled_img.shape[0]:
                filtered_contours.append(cnt)
                cv2.drawContours(filled_img, [cnt], -1, 255, thickness=cv2.FILLED)

    small_kernel = np.ones((4, 4), np.uint8)
    filled_img = cv2.dilate(filled_img, small_kernel, iterations=3)

    final_contours, _ = cv2.findContours(filled_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(final_contours)

def main(dataset_folder):
    duck_count_df = pd.read_csv(f'{dataset_folder}/duck_count.csv')
    true_counts = duck_count_df.set_index('picture')['ducks']

    predicted_counts = []
    for i in range(1, 11):
        image_path = f'{dataset_folder}/picture_{i}.jpg'
        predicted_count = count_ducks_with_filled_contours(image_path)
        predicted_counts.append(predicted_count)

    predicted_counts_series = pd.Series(predicted_counts, index=true_counts.index)
    mae = np.mean(np.abs(predicted_counts_series - true_counts))
    print(f"{mae}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python script.py <dataset_folder>")
    main(sys.argv[1])
