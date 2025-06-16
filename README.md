# 🏥 Hospital Face Mask Detection with Live Alert System 😷

A real-time face mask detection system built for hospital environments using OpenCV, TensorFlow, and Haar Cascades. This project uses annotated face mask data to train a Convolutional Neural Network (CNN) that can distinguish between masked and unmasked individuals via webcam and raise alerts.

---

## 🚀 Features

- ✅ Real-time face detection via webcam  
- ✅ Mask vs. No-Mask classification using CNN  
- ✅ Alerts when someone is detected **without** a mask  
- ✅ Trained on real images with XML annotations  
- ✅ Easy to deploy on any Windows/Linux laptop with a webcam  

---

## 📁 Folder Structure
python_Folder/
├── annotations/ # Pascal VOC XML annotations
├── images/ # Raw face images
├── haarcascade_frontalface_default.xml # Haar Cascade for face detection
├── parse_data.py # Script to parse and prepare dataset
├── train_model.py # Script to train CNN model
├── mask_model.h5 # Saved model after training
├── file1.py # Real-time webcam detection

dataset:
  name: Face Mask Detection
  source: Kaggle
  kaggle_url: "https://www.kaggle.com/datasets/andrewmvd/face-mask-detection"
  download_instructions: |
    - Go to the URL above
    - Log in with your Kaggle account
    - Click "Download All"
    - Extract the zip, and place:
       - `images/` folder inside `python_Folder/`
       - `annotations/` folder inside `python_Folder/`

requirements:
  install_command: pip install opencv-python tensorflow lxml tqdm scikit-learn

tech_stack:
  - Python
  - TensorFlow/Keras
  - OpenCV
  - Haar Cascade Classifier
  - Pascal VOC Annotation Parsing

applications:
  - Hospitals & Clinics
  - Offices & Public Institutions
  - Entrance Monitoring Systems
  - Smart Surveillance

future_improvements:
  - Integrate MobileNetV2 for faster inference
  - Add email/SMS alerts using Twilio
  - Deploy via Flask dashboard or Raspberry Pi

developer:
  - name: Parks RPK
  - title: Computer Science @ Binghamton University
  - skills: "💻 Machine Learning • Web Dev • Systems • Marathoner 🏃‍♂️"
  - linkedin: "https://www.linkedin.com/in/parks-rpk-8479a3350/"


