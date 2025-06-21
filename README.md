# 🛡️ Ransomware Detection and Prevention - G05-S13

This repository contains the final project for the course **NT230 – Malware’s Modus Operandi** at **University of Information Technology (UIT), VNU-HCM**, conducted in Semester 2, School Year 2024–2025.

## 📌 Project Title
**Explainable deep learning-based ransomware detection using dynamic analysis**

## 👥 Team Members (Group G05)
- Hồ Hoàng Diệp - 22520249  
- Nguyễn Đặng Nguyên Khang - 22520617  
- Trần Vỹ Khang - 22520628  

## 🎯 Objectives
- Detect ransomware using **dynamic analysis** (API Calls, DLLs, Mutexes via Cuckoo Sandbox).
- Train a lightweight yet effective **2-layer CNN** to classify malware.
- Use **Explainable AI (XAI)** techniques like **LIME** and **SHAP** to interpret the model's decisions.

## 🧪 Methodology
- Run samples in **Cuckoo Sandbox** to collect behavioral features.
- Extract and combine ordered sequences of API calls, DLLs, and mutexes.
- Feed combined vectors into a **2-layer 1D-CNN** for classification.
- Provide **explainability** using LIME (for local explanations) and SHAP (for global importance).

## ⚙️ Implementation Details
- **Programming Language**: Python 3.11
- **Core Libraries**: TensorFlow, LIME, SHAP, scikit-learn, matplotlib, Streamlit, Flask
- **Model architecture**: Embedding → Conv1D (128) → MaxPooling → Dropout → Conv1D (64) → MaxPooling → Dropout → Dense(64) → Dropout → Softmax(2)
- **Backend**: Flask REST API
- **Frontend**: Streamlit-based web interface
- **Explainability**:
  - **LIME**: visualizes important features affecting each prediction.
  - **SHAP**: global interpretability across datasets.

## 🖥️ Experimental Setup
- **Cuckoo Sandbox**: running on Ubuntu 20.04 with Windows 7 VM for dynamic analysis
- **Development machine**:
  - CPU: Intel Core i7-11390H
  - RAM: 16GB
- **Dataset sources**: theZoo, AnyRun, MarauderMap, SOREL-20M

## 📊 Results
- **Ransomware vs. Benign**:
  - **Accuracy**: 99.10%
  - **TPR**: 98.04%
  - **FPR**: 0.00%
  - **F1-Score**: 0.9901

- **Ransomware vs. Malware**:
  - **Accuracy**: 94.12%
  - **TPR**: 94.12%
  - **FPR**: 5.88%
  - **F1-Score**: 0.9412

## 📈 Feature Design and Input Encoding
- Max API length: **500**
- Max DLLs: **10**
- Max Mutexes: **10**
- Concatenation order: `API || DLL || Mutex`
- Tokenization: integer mapping, embedding for semantic learning

## 📚 Key Techniques
- **Dynamic Analysis** via Cuckoo Sandbox
- **Feature extraction** from JSON reports (API, DLL, Mutex)
- **2-layer 1D-CNN** for sequence learning and classification
- **Explainability** using LIME (local) and SHAP (global)

## 🌐 Demo Interface Pipeline
- Upload file (.exe / .json / attribute file)
- Analyze in Cuckoo (if execute), extract features
- Predict with trained 2L-CNN
- Visualize LIME chart on top features

## 🚧 Challenges Encountered
- Ransomware samples misclassified in malware datasets
- Behavioral similarity between malware types
- Resource-heavy SHAP explanation on local machine

## 🚀 Future Directions
- Integrate static analysis (e.g. PE header, opcode)
- Improve runtime with lighter model (e.g. 1D-MobileNet)
- Real-time protection on endpoint/gateway
- Visual dashboard for SHAP and LIME
- Integrate auto-response and recovery (e.g. isolate, alert, rollback)

## 🌐 Demo
A simple web demo was built for uploading a file and classifying whether it is ransomware using the trained model and LIME explanations.
**Video demo**: [Google Drive Link](https://drive.google.com/drive/folders/1QA2LLAGzTbHEY7NUwAOKfUqxsvwp0gNu?usp=sharing)

## 📄 Files
- [`Slide-G05-S13.pdf`](./Slide-G05-S13.pdf) — Project presentation slides
- [`Research_Poster-G05-S13.pdf`](./Research_Poster-G05-S13.pdf) — Academic research poster
- [`[NT230.P21.ANTN]-Project_Final_Nhom05_S13.pdf`](./[NT230.P21.ANTN]-Project_Final_Nhom05_S13.pdf) — Detail report

## References
Sibel Gulmez, Arzu Gorgulu Kakisim, and Ibrahim Sogukpinar. 2024. XRan: Explainable deep learning-based ransomware detection using dynamic analysis. Comput. Secur. 139, C (Apr 2024). https://doi.org/10.1016/j.cose.2024.103703

## 🔖 License
© UIT InSecLab, NT230 Course — For academic use only.

