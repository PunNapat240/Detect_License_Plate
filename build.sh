#!/bin/bash
# ติดตั้ง Tesseract OCR พร้อมภาษาไทย
apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-tha

# ตรวจสอบว่า Tesseract ติดตั้งสำเร็จ
which tesseract
tesseract -v

# ติดตั้ง Python dependencies
pip install -r requirements.txt

# รันเซิร์ฟเวอร์ด้วย Gunicorn
# gunicorn server:app
