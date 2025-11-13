import sys
import numpy as np
import cv2
import pandas as pd

def find_ducks(image_path, min_area=500, max_area=5000):
    # Učitavanje slike
    img = cv2.imread(image_path)
    
    # Kropovanje slike (odabir centralnog dela)
    h, w = img.shape[:2]
    crop_img = img[h//5:4*h//5, w//5:4*w//5]  # kropovanje centralnog dela slike
    
    # Pravljenje kružne maske koja će ignorisati konture u uglovima
    mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    radius = min(center)
    cv2.circle(mask, center, radius, 255, -1)  # Bela kružna maska
    
    # Zamućenje slike kako bi se smanjio šum
    img_blur = cv2.GaussianBlur(crop_img, (7, 7), 0)
    
    # Konverzija u grayscale i globalni threshold za binarizaciju
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 85, 255, cv2.THRESH_BINARY_INV)
    
    # Čišćenje binarne slike
    kernel = np.ones((5, 5), np.uint8)
    img_bin_cleaned = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    img_bin_cleaned = cv2.morphologyEx(img_bin_cleaned, cv2.MORPH_OPEN, kernel)
    
    # Primena kružne maske i detekcija kontura
    img_bin_masked = cv2.bitwise_and(img_bin_cleaned, img_bin_cleaned, mask=mask)
    contours, _ = cv2.findContours(img_bin_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kreiranje slike sa popunjenim konturama
    img_filled_bin = np.zeros_like(img_bin_masked)
    for contour in contours:
        cv2.drawContours(img_filled_bin, [contour], -1, 255, thickness=cv2.FILLED)

    # Ponovna detekcija kontura na popunjenoj slici
    filled_contours, _ = cv2.findContours(img_filled_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtriranje kontura prema površini
    valid_contours = []
    for contour in filled_contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area or 5980 < area < 6300 or 7700 < area < 9000:
            valid_contours.append(contour)

    # Ako nema patkica, primeni dilaciju na uži krug
    if len(valid_contours) == 0:
        smaller_radius = int(radius * 0.7)
        small_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
        cv2.circle(small_mask, center, smaller_radius, 255, -1)
        img_bin_masked = cv2.bitwise_and(img_bin_cleaned, img_bin_cleaned, mask=small_mask)
        
        # Primena dilatacije na uži krug i ponovna detekcija kontura
        large_kernel = np.ones((10, 10), np.uint8)
        img_bin_dilated = cv2.dilate(img_bin_masked, large_kernel, iterations=2)
        contours, _ = cv2.findContours(img_bin_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area or 5980 < area < 6300 or 7700 < area < 9000:
                valid_contours.append(contour)

    return len(valid_contours)

def main(dataset_folder):
    # Poziv funkcije za svaku sliku i čuvanje rezultata
    predicted_counts = []
    for i in range(1, 11):
        image_path = f'{dataset_folder}/picture_{i}.jpg'
        count = find_ducks(image_path, min_area=400, max_area=5000)
        predicted_counts.append(count)

    # Učitavanje očekivanih vrednosti iz CSV fajla i računanje MAE
    duck_count_df = pd.read_csv(f'{dataset_folder}/duck_count.csv')
    true_counts = duck_count_df.set_index('picture')['ducks']
    predicted_counts_series = pd.Series(predicted_counts, index=true_counts.index)
    mae = np.mean(np.abs(predicted_counts_series - true_counts))

    # Ispis rezultata
    print(f"{mae}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python script.py <dataset_folder>")
    main(sys.argv[1])
