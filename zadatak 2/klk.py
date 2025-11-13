import cv2
import pandas as pd
import numpy as np
import os
import sys

def count_and_evaluate_buzzy_beetles(video_path, track_line=450, min_contour_area=1000, max_contour_area=2000, distance_threshold=40):
    cap = cv2.VideoCapture(video_path)
    count = 0
    tracked_objects = []  # Lista za praćenje objekata (x, y)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Kropovanje na srednji kvadrat
        h, w, _ = frame.shape
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2 + 200
        frame = frame[start_y:start_y+size - 100, start_x:start_x+size]

        # Konverzija u HSV i maskiranje tamno plave boje
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([85, 80, 45])  # Donja granica za tamno plavu
        upper_blue = np.array([140, 255, 255])  # Gornja granica za tamno plavu
        mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # Pronalaženje kontura
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_contour_area <= area <= max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                # Provera da li je objekat već praćen
                is_new_object = True
                for (prev_x, prev_y) in tracked_objects:
                    if np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) < distance_threshold:
                        is_new_object = False
                        break

                # Ako je novi objekat i prelazi liniju, dodaj ga u praćenje
                if is_new_object and center_x > track_line:
                    tracked_objects.append((center_x, center_y))
                    count += 1

    cap.release()
    return count

def main(dataset_folder):
    ground_truth_path = os.path.join(dataset_folder, 'buzzy_beetle_count.csv')
    videos = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.mp4')]

    # Učitavanje tačnih vrednosti
    ground_truth = pd.read_csv(ground_truth_path)

    # Procesiranje svakog videa
    predicted_counts = []
    for video_path in sorted(videos):
        video_name = os.path.basename(video_path)
        count = count_and_evaluate_buzzy_beetles(video_path)
        predicted_counts.append({'video': video_name, 'predicted_count': count})

    predicted_df = pd.DataFrame(predicted_counts)
    result_df = pd.merge(ground_truth, predicted_df, on='video')
    result_df['error'] = abs(result_df['count'] - result_df['predicted_count'])
    mae = result_df['error'].mean()

    # Ispis rezultata (samo broj)
    print(f"{mae:.1f}")

if __name__ == "__main__":
    dataset_folder = sys.argv[1]  # Prvi argument komandne linije
    main(dataset_folder)
