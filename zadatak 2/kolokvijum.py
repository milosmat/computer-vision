import cv2
import numpy as np
import pandas as pd
import sys

def count_blue_objects_crossing_center(video_path, show_frames=False, roi=None, skip_frames=3):
    cap = cv2.VideoCapture(video_path)
    beetle_count = 0
    counted_ids = set()
    skip_counter = 0

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    ret, first_frame = cap.read()
    if not ret:
        return 0

    if roi is None:
        roi = (0, 0, first_frame.shape[1], first_frame.shape[0])

    x_roi, y_roi, w_roi, h_roi = roi

    lower_blue = np.array([60, 110, 150])
    upper_blue = np.array([82, 160, 172])

    center_x = w_roi // 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if skip_counter > 0:
            skip_counter -= 1
            continue

        cropped_frame = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        fgmask = fgbg.apply(cropped_frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        #_, fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if 1000 < cv2.contourArea(cnt) < 4000:
                x, y, w, h = cv2.boundingRect(cnt)
                obj_id = (x, y, w, h)
                center_of_object = x + w // 2 - 120

                if center_x - 6 <= center_of_object <= center_x + 6:
                    roi_hsv = cv2.cvtColor(cropped_frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
                    mean_hsv = cv2.mean(roi_hsv)[:3]

                    if lower_blue[0] <= mean_hsv[0] <= upper_blue[0] and \
                       lower_blue[1] <= mean_hsv[1] <= upper_blue[1] and \
                       lower_blue[2] <= mean_hsv[2] <= upper_blue[2]:
                        counted_ids.add(obj_id)
                        beetle_count += 1
                        skip_counter = skip_frames
                        cv2.rectangle(cropped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.imshow('Slika beetl-a',cropped_frame)
                        cv2.waitKey(0)

            elif 9000 < cv2.contourArea(cnt) < 12000 or 13000 < cv2.contourArea(cnt) < 40000: 
                x, y, w, h = cv2.boundingRect(cnt)
                step_x = max(w // 3, 1) 
                step_y = max(h // 3, 1) 

                for i in range(0, w, step_x):
                    for j in range(0, h, step_y):
                        sub_x = x + i
                        sub_y = y + j
                        sub_w = min(step_x, w - i)
                        sub_h = min(step_y, h - j)

                        sub_obj_id = (sub_x, sub_y, sub_w, sub_h)

                        center_of_object = sub_x + sub_w // 2 + 110

                        if center_x - 15 <= center_of_object <= center_x + 15 and sub_obj_id not in counted_ids:
                            roi_hsv = cv2.cvtColor(cropped_frame[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w], cv2.COLOR_BGR2HSV)
                            mean_hsv = cv2.mean(roi_hsv)[:3]

                            if 56 <= mean_hsv[0] <= 82 and \
                            (110 <= mean_hsv[1] <= 130 or 140 <= mean_hsv[1] <= 160) and \
                            150 <= mean_hsv[2] <= 170:
                                counted_ids.add(sub_obj_id) 
                                beetle_count += 1
                                skip_counter = 40
                                cv2.rectangle(cropped_frame, (sub_x, sub_y), (sub_x+step_x, sub_y+step_y), (0,255,0), 2)
                                cv2.imshow('slika pravougaonik', cropped_frame)
                                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    return beetle_count

def main(dataset_folder):
    ground_truth_df = pd.read_csv(f"{dataset_folder}/buzzy_beetle_count.csv")
    ground_truth_series = ground_truth_df.set_index('video')['count']

    predicted_counts = []
    for i in range(1, 11):
        video_path = f"{dataset_folder}/video_{i}.mp4"
        roi = (400, 250, 500, 320)
        beetle_count = count_blue_objects_crossing_center(video_path, show_frames=False, roi=roi)
        predicted_counts.append(beetle_count)

    predicted_counts_series = pd.Series(predicted_counts, index=ground_truth_series.index)
    mae = np.mean(np.abs(predicted_counts_series - ground_truth_series))
    print(f"{mae:.1f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python script.py <dataset_folder>")
    main(sys.argv[1])