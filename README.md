# 🛡️ Ransomware Detection and Prevention - G05-S13

This repository contains the final project for the course **NT230 – Malware’s Modus Operandi** at **University of Information Technology (UIT), VNU-HCM**, conducted in Semester 2, School Year 2024–2025.

## 📌 Project Title
**Explainable deep learning-based ransomware detection using dynamic analysis**

## References
Sibel Gulmez, Arzu Gorgulu Kakisim, and Ibrahim Sogukpinar. 2024. XRan: Explainable deep learning-based ransomware detection using dynamic analysis. Comput. Secur. 139, C (Apr 2024). https://doi.org/10.1016/j.cose.2024.103703

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
- Provide **explainability** using LIME (local) and SHAP (global).

## 📊 Results
- **Ransomware vs. Benign**:
  - Accuracy: **99.10%**
  - TPR: **98.04%**
  - FPR: **0.00%**
  - F1-Score: **0.9901**

- **Ransomware vs. Malware**:
  - Accuracy: **94.12%** (2L-CNN)
  - FPR: **0.0588**
  - 2L-CNN performance matches or exceeds traditional models (Random Forest, Decision Tree)

## 🌐 Demo
A simple web demo was built for uploading a file and classifying whether it is ransomware using the trained model and LIME explanations.

## 📄 Files
- [`Slide-G05-S13.pdf`](./Slide-G05-S13.pdf) — Project presentation slides
- [`Research_Poster-G05-S13.pdf`](./Research_Poster-G05-S13.pdf) — Academic research poster
- [`[NT230.P21.ANTN]-Project_Final_Nhom05_S13.pdf`](./[NT230.P21.ANTN]-Project_Final_Nhom05_S13.pdf) - Detail report

## 🔖 License
© UIT InSecLab, NT230 Course — For academic use only.

