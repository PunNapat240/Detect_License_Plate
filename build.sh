#!/bin/bash
apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-tha
pip install -r requirements.txt
gunicorn server:app
