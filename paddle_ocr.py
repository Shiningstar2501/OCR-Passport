from paddleocr import PaddleOCR
import cv2
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def perform_ocr(image_path):
    img = cv2.imread(image_path)
    results = ocr.ocr(img, cls=True)
    extracted_texts = [line[1][0] for line in results[0]]
    return extracted_texts
