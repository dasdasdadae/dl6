import os
import sys
import cv2
import joblib
import numpy as np
from ultralytics import YOLO

import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger("ultralytics").setLevel(logging.WARNING)

def calc_stats(arr):
    arr = np.array(arr)
    if len(arr) == 0:
        return [0, 0, 0, 0]
    return [
        np.mean(arr),
        np.var(arr),
        np.ptp(arr),
        np.mean(np.abs(np.diff(arr))) if len(arr) > 1 else 0
    ]

def extract_features(folder_path, pose_model):
    valid_ext = ('.png', '.jpg', '.jpeg')
    image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)])

    # Защита от пустых папок
    if len(image_paths) == 0:
        raise ValueError(f"В папке {folder_path} не найдено ни одной картинки!")

    # Если картинок меньше 8 - дублируем последнюю, если больше - обрезаем
    if len(image_paths) < 8:
        image_paths = image_paths + [image_paths[-1]] * (8 - len(image_paths))
    elif len(image_paths) > 8:
        image_paths = image_paths[:8]

    kp_seq = []
    conf_seq = []

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        res = pose_model(img, verbose=False)

        if len(res[0].boxes) > 0:
            data = res[0].keypoints.data[0].cpu().numpy()
            kp = data[:, :2]
            conf = data[:, 2]

            box = res[0].boxes.xyxy[0].cpu().numpy()
            w, h = box[2] - box[0], box[3] - box[1]
            if w > 0 and h > 0:
                kp[:, 0] = (kp[:, 0] - box[0]) / w
                kp[:, 1] = (kp[:, 1] - box[1]) / h
        else:
            if i == 0:
                kp, conf = np.zeros((17, 2)), np.zeros(17)
            else:
                kp, conf = kp_seq[-1].copy(), conf_seq[-1].copy()

        kp_seq.append(kp)
        conf_seq.append(conf)

    kp_seq = np.array(kp_seq)
    conf_seq = np.array(conf_seq)

    CONF_THRESHOLD = 0.4
    for i in range(1, len(kp_seq)):
        low_conf_mask = conf_seq[i] < CONF_THRESHOLD
        kp_seq[i, low_conf_mask, :] = kp_seq[i-1, low_conf_mask, :]

    features = []

    # А. Ступни (16 признаков)
    for foot_idx in [15, 16]:
        features.extend(calc_stats(kp_seq[:, foot_idx, 0]))
        features.extend(calc_stats(kp_seq[:, foot_idx, 1]))

    # Б. Центр масс (4 признака)
    center_x = (kp_seq[:, 11, 0] + kp_seq[:, 12, 0]) / 2
    center_y = (kp_seq[:, 11, 1] + kp_seq[:, 12, 1]) / 2
    features.extend([np.mean(center_x), np.var(center_x)])
    features.extend([np.mean(center_y), np.var(center_y)])

    # В. Руки (8 признаков)
    for wrist, shoulder in [(9, 5), (10, 6)]:
        dist = np.linalg.norm(kp_seq[:, wrist, :] - kp_seq[:, shoulder, :], axis=1)
        features.extend(calc_stats(dist))

    # Г. Расстояние от кисти до носа (4 признака)
    for wrist in [9, 10]:
        dist_to_nose = np.linalg.norm(kp_seq[:, wrist, :] - kp_seq[:, 0, :], axis=1)
        features.extend([np.mean(dist_to_nose), np.min(dist_to_nose)])

    # Д. Ширина плеч (1 признак)
    shoulder_width = np.linalg.norm(kp_seq[:, 5, :] - kp_seq[:, 6, :], axis=1)
    features.append(np.mean(shoulder_width))

    # Е. Дисперсия расстояния между ступнями (1 признак)
    feet_dist = np.linalg.norm(kp_seq[:, 15, :] - kp_seq[:, 16, :], axis=1)
    features.append(np.var(feet_dist))

    return np.array(features).reshape(1, -1)

def main():
    if len(sys.argv) < 2:
        print("Ошибка: Укажите путь к папке с изображениями")
        sys.exit(1)

    target_folder = sys.argv[1]

    # 1. Загружаем модели
    pose_model = YOLO('yolov8n-pose.pt')
    clf_model = joblib.load('best_model.pkl')

    # 2. Вытаскиваем фичи (теперь их ровно 34)
    features = extract_features(target_folder, pose_model)

    # 3. Делаем предикт
    pred_idx = int(clf_model.predict(features)[0])

    classes = ['inaction', 'move', 'work']
    print(classes[pred_idx])

if __name__ == "__main__":
    main()
