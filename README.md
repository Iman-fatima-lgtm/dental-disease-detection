# 🦷 Dental Disease Detection using YOLOv8

This project implements a **Dental Disease Detection System** using **YOLOv8**, a state-of-the-art object detection model. It is developed in **Google Colab** and detects dental abnormalities or diseases from intraoral or panoramic dental images. The goal is to assist dentists and radiologists in identifying issues such as cavities, plaque, root infections, and more through real-time AI-driven image analysis.

---

## 📄 Project Summary

Manual diagnosis of dental diseases through X-rays or oral imagery is time-consuming, prone to error, and heavily reliant on expert knowledge. This system leverages **YOLOv8** to automate the detection of dental anomalies in a fast, accurate, and consistent manner.

The system is deployed using **Gradio**, enabling users to upload dental images via a browser and receive annotated output images showing detected disease regions.

---

## 🎯 Objectives

- Automate detection of dental diseases from images
- Provide bounding box predictions for common dental issues
- Deliver real-time detection via a lightweight Gradio app
- Improve diagnostic support in dental clinics and academic settings

---

## 📁 Dataset Structure

The dataset should follow YOLOv8's expected format:



data/
├── images/
│ ├── train/
│ └── val/
├── labels/
│ ├── train/
│ └── val/
└── dataset.yaml


names:
  - Calculus
  - Plaque
  - caries
  - gingivitis
  - lessions
  - tooth discoloration
  - ulcer
  - xerostomia
nc: 8
roboflow:
  license: CC BY 4.0
  project: oral-detector-5
  url: https://universe.roboflow.com/clients-mpvn2/oral-detector-5/dataset/2
  version: 2
  workspace: clients-mpvn2
train: /content/ORAL-DETECTOR-5-2/train/images
val: /content/ORAL-DETECTOR-5-2/valid/images
test: /content/ORAL-DETECTOR-5-2/test/images
🛠️ Tools & Technologies
Purpose	Tools Used
Programming	Python 3.11
Deep Learning	YOLOv8 (Ultralytics), PyTorch
Image Processing	OpenCV, NumPy
UI & Deployment	Gradio, Google Colab
Version Control	Git, GitHub

🔧 Installation
Install dependencies in Google Colab or your local machine:

bash
Copy
Edit
pip install ultralytics opencv-python gradio numpy
🧪 Training the Model
python
Copy
Edit
from ultralytics import YOLO

# Load YOLOv8n or custom model
model = YOLO("yolov8n.yaml")

# Train
model.train(data="data/dataset.yaml", epochs=50, imgsz=640)
🔍 Inference (Tumor Detection)
python
Copy
Edit
model = YOLO("runs/detect/train/weights/best.pt")
results = model("sample_mri.jpg", save=True)
🌐 Gradio Web Interface
python
Copy
Edit
import gradio as gr
from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

def detect(image):
    results = model(image)
    return results[0].plot()

gr.Interface(fn=detect, inputs="image", outputs="image", title="Dental Disease Detector").launch(share=True)
🧪 Testing Scenarios
Test Case	Description
✅ Image Upload	Upload valid  image and view detection result
✅ Invalid Image	Handle unsupported or corrupted files gracefully
✅ Accuracy Check	Verify bounding box quality & model confidence

🔐 Ethical & Legal Considerations
Images must be anonymized

Tool is assistive, not a replacement for a doctor

Compliant with ethical AI and data protection laws

📈 Future Enhancements
Expand to 3D MRI and DICOM formats

Add Disease classification (e.g.,Calculus
  - Plaque
  - caries
  - gingivitis
  - lessions
  - tooth discoloration
  - ulcer
  - xerostomia)

Integrate with PACS/EMR hospital systems

👩‍⚕️ Stakeholders
Medical professionals (radiologists, doctors)

Hospitals & diagnostic centers

Patients (via telemedicine)

Researchers & students in computer vision

🙌 Acknowledgements
Ultralytics YOLOv8

University of Agriculture Faisalabad, Dept. of CS

Project by Iman Fatima
BS Software Engineering (2021-ag-8053)

📄 References
Jiang et al., YOLOv5-based brain tumor detection, Computers in Biology and Medicine

Ultralytics YOLOv8 Docs: https://docs.ultralytics.com

Gradio: https://www.gradio.app
