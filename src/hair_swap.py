import cv2
import numpy as np
import torch
import os
import mediapipe as mp
from u2net import U2NET

# Конфигурация
MODEL_PATH = "u2net.pth"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
TARGET_SIZE = 512  # Размер должен быть кратен 16


def load_model():
    """Загрузка предобученной модели U²-Net"""
    model = U2NET()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл весов {MODEL_PATH} не найден!")
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model


def process_mask(mask, kernel_size=5):
    """Постобработка маски"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def segment_hair(model, image, threshold=0.7):
    """Сегментация волос"""
    h, w = image.shape[:2]
    img = cv2.resize(image, (320, 320))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        mask = model(img_tensor)[0].squeeze().numpy()

    mask = cv2.resize(mask, (w, h))
    mask = (mask > threshold).astype(np.uint8) * 255
    return process_mask(mask)


def safe_resize(img, size):
    """Изменение размера с сохранением пропорций"""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def detect_face_center(image):
    """Определение центра лица с защитой от ошибок"""
    try:
        mp_face = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        results = mp_face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return None

        h, w = image.shape[:2]
        bbox = results.detections[0].location_data.relative_bounding_box

        # Расчет координат с защитой
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        width = min(w - x, int(bbox.width * w))
        height = min(h - y, int(bbox.height * h))

        cx = x + width // 2
        cy = y + height // 3  # Смещение к области волос

        return (
            np.clip(cx, 0, w - 1).item(),
            np.clip(cy, 0, h - 1).item()
        )
    except Exception:
        return None


def blend_images(src, dst, mask):
    """Альтернативное смешивание"""
    mask = mask.astype(np.float32) / 255.0
    blended = dst * (1 - mask[..., np.newaxis]) + src * mask[..., np.newaxis]
    return blended.astype(np.uint8)


def swap_hair(source_path, target_path):
    """Основная функция замены прически"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Загрузка и предобработка
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        if source is None or target is None:
            raise ValueError("Ошибка загрузки изображений!")

        # Приведение к целевому размеру
        source = safe_resize(source, TARGET_SIZE)
        target = safe_resize(target, TARGET_SIZE)
        target = cv2.resize(target, (source.shape[1], source.shape[0]))

        # Сегментация
        model = load_model()
        source_mask = segment_hair(model, source)
        target_mask = segment_hair(model, target)

        # Валидация масок
        cv2.imwrite(os.path.join(OUTPUT_DIR, "1_source_mask.jpg"), source_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "2_target_mask.jpg"), target_mask)

        if cv2.countNonZero(target_mask) < 1000:
            raise ValueError("Маска прически слишком маленькая!")

        # Очистка исходного изображения
        source_cleaned = cv2.inpaint(source, source_mask, 5, cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "3_cleaned.jpg"), source_cleaned)

        # Определение центра
        center = detect_face_center(source_cleaned) or (
            source.shape[1] // 2,
            source.shape[0] // 4
        )
        center = (
            np.clip(center[0], 0, source.shape[1] - 1),
            np.clip(center[1], 0, source.shape[0] - 1)
        )

        # Визуализация центра
        debug_img = source_cleaned.copy()
        cv2.circle(debug_img, center, 15, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "4_center.jpg"), debug_img)

        # Попытка 1: Использование seamlessClone
        try:
            result = cv2.seamlessClone(
                target,
                source_cleaned,
                target_mask,
                center,
                cv2.MIXED_CLONE
            )
        except Exception as e:
            print(f"Ошибка seamlessClone: {str(e)}")
            # Попытка 2: Ручное смешивание
            result = blend_images(target, source_cleaned, target_mask)

        # Сохранение результатов
        cv2.imwrite(os.path.join(OUTPUT_DIR, "5_result.jpg"), result)
        print("Успешно! Результаты в папке output:")
        print("- 1_source_mask.jpg\n- 2_target_mask.jpg\n- 3_cleaned.jpg")
        print("- 4_center.jpg\n- 5_result.jpg")

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    swap_hair(
        os.path.join(INPUT_DIR, "source.jpg"),
        os.path.join(INPUT_DIR, "target.jpg")
    )