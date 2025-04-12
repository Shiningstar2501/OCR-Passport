from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(use_angle_cls=False, lang='en')

def resize_image(image, max_size=1024):
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / float(max(height, width))
        return cv2.resize(image, (int(width * scale), int(height * scale)))
    return image

def perform_ocr(image_path):
    img = cv2.imread(image_path)
    img = resize_image(img)  # ✅ Resize to limit RAM usage
    results = ocr.ocr(img, cls=False)
    return [line[1][0] for line in results[0]]
