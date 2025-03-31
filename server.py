import numpy as np
import cv2
import pytesseract
from flask import Flask, request, jsonify, render_template
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# กำหนด path ของ Tesseract OCR (แก้ไขให้ตรงกับเครื่องของคุณ)
# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def apply_gamma_correction(image, gamma=1.5):
    """ ปรับค่า Gamma Correction เพื่อเพิ่มความคมชัดของภาพ """
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, gamma_table)

def enhance_contrast(image):
    """ เพิ่ม Contrast ของภาพโดยใช้ CLAHE (Contrast Limited Adaptive Histogram Equalization) """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

@app.route('/')
def home():
    return render_template("index.html", imageLatest_url="/static/uploads/latest.jpg", imageDetect_url="/static/uploads/detect.jpg")

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # 📌 รับข้อมูลภาพ Grayscale จาก ESP32
        image_data = request.data
        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        width, height = 640, 480  # ขนาดของภาพที่ ESP32-CAM ส่งมา
        img_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width))

        # 🎨 ปรับปรุงภาพให้เหมาะกับ OCR
        img_corrected = apply_gamma_correction(img_array, gamma=1.5)  # เพิ่ม contrast
        img_enhanced = enhance_contrast(img_corrected)

        # 💾 บันทึกเป็น JPEG
        image_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")
        Image.fromarray(img_enhanced, mode="L").save(image_path, "JPEG")

        print("✅ ภาพถูกบันทึกสำเร็จ!")

        # 🔍 ตรวจจับป้ายทะเบียน
        license_plate_text = detect_license_plate(img_enhanced)
        print("เลขทะเบียนที่ตรวจจับได้:", license_plate_text)

        return jsonify({"message": "Complete", "license_plate": license_plate_text}), 200

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route('/upload_test', methods=['POST'])
def upload_test():
    """ 📸 ฟังก์ชันสำหรับอัปโหลดภาพ JPG เพื่อตรวจสอบการตรวจจับป้ายทะเบียน """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # ตรวจสอบว่าเป็นไฟล์ JPG หรือไม่
        if not file.filename.lower().endswith(('.jpg', '.jpeg')):
            return jsonify({"error": "Only JPG files are allowed"}), 400

        # 📥 อ่านภาพที่อัปโหลด
        image_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")
        file.save(image_path)
        print(f"✅ อัปโหลดไฟล์สำเร็จ: {image_path}")

        # 📷 โหลดภาพ
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Failed to read image"}), 500

        # 🎨 แปลงภาพเป็น Grayscale และปรับปรุงคุณภาพ
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_corrected = apply_gamma_correction(gray, gamma=1.5)
        img_enhanced = enhance_contrast(img_corrected)

        # 🔍 ตรวจจับป้ายทะเบียน
        license_plate_text = detect_license_plate(img_enhanced)
        print("เลขทะเบียนที่ตรวจจับได้:", license_plate_text)

        return jsonify({"message": "Complete", "license_plate": license_plate_text}), 200

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

def detect_license_plate(image):
    """ ตรวจจับป้ายทะเบียนจากภาพ Grayscale โดยใช้ Canny และ Tesseract """
    try:
        # ตรวจสอบว่าภาพมี 1 channel (Grayscale)
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 🔹 ลด noise ด้วย bilateral filter
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # 🔹 ตรวจจับขอบด้วย Canny
        edged = cv2.Canny(gray, 30, 200)

        # 🔹 หาคอนทัวร์จากภาพขอบ
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        license_plate_contour = None

        # 🔹 ค้นหาคอนทัวร์ที่เป็นสี่เหลี่ยม
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                license_plate_contour = approx
                break

        if license_plate_contour is None:
            return "ไม่พบป้ายทะเบียน"

        # 🔹 สร้าง mask สำหรับตัดเฉพาะบริเวณป้ายทะเบียน
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [license_plate_contour], 0, 255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)
        image_path = os.path.join(UPLOAD_FOLDER, "detect.jpg")
        Image.fromarray(new_image, mode="L").save(image_path, "JPEG")

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped = gray[topx:bottomx+1, topy:bottomy+1]

        # 🔹 OCR อ่านตัวอักษรจากป้ายทะเบียน
        custom_config = r'--oem 3 --psm 6 tessedit_char_whitelist=0123456789กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
        text = pytesseract.image_to_string(cropped, lang='tha', config=custom_config)


        print("🚗 เลขทะเบียนที่ตรวจจับได้:", text.strip())
        return text.strip()

    except Exception as e:
        print(f"⚠️ ตรวจจับป้ายทะเบียนล้มเหลว: {str(e)}")
        return "ตรวจจับไม่สำเร็จ"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
