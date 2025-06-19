# 🧠 Brain Tumor Detection using YOLOv8

This project implements a **Brain Tumor Detection System** using **YOLOv8**, a deep learning-based object detection model. The system is built and tested in **Google Colab** and utilizes **MRI scans** to detect and localize tumors in real time. It’s designed as an assistive tool for medical professionals to improve diagnostic accuracy and speed.

---

## 🚀 Project Summary

Brain tumors are a critical health threat requiring timely and accurate diagnosis. Manual inspection of MRI images is time-consuming, subjective, and error-prone. This project addresses those limitations by applying YOLOv8 to automatically detect tumor regions in brain MRI scans, using a fast and accurate computer vision pipeline.

The system is accessible via a **Gradio web interface**, allowing users to upload an image and receive annotated predictions with tumor bounding boxes.

---

## 🧠 Project Goals

- Automate brain tumor detection using MRI images
- Achieve real-time inference with YOLOv8
- Design a user-friendly interface using Gradio
- Improve detection accuracy while reducing false negatives
- Enable accessibility for low-resource settings

---

## 📁 Dataset Format

Dataset is expected in **YOLOv8-compatible format**:

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

gr.Interface(fn=detect, inputs="image", outputs="image", title="Brain Tumor Detector").launch(share=True)
🧪 Testing Scenarios
Test Case	Description
✅ Image Upload	Upload valid MRI image and view detection result
✅ Invalid Image	Handle unsupported or corrupted files gracefully
✅ Accuracy Check	Verify bounding box quality & model confidence

🔐 Ethical & Legal Considerations
Images must be anonymized

Tool is assistive, not a replacement for a doctor

Compliant with ethical AI and data protection laws

📈 Future Enhancements
Expand to 3D MRI and DICOM formats

Add tumor classification (e.g., glioma, meningioma)

Improve detection of small/silent tumors

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
