ยินดีต้อนรับทุกคนสู่ โปรเจคEmguard
โปรเจคนี้คือโปรเจคในการ Detect ไวรัสคอมพิวเตอร์โดยเพียงแค่อัปไฟล์ 

Data set of this project
Dataset: EMBER2024 (https://huggingface.co/datasets/joyce8/EMBER2024)
(เดิมใช้ EMBER 2018: https://ember.elastic.co/ember_dataset_2018_2.tar.bz2)

Train by Neuron network

Use with this link

https://virusdetect.streamlit.app/

This is my blog link https://medium.com/@jinnathit/emguard-virus-file-detection-5f43a6f1d25f

If  you have any question Please email to me : jinnathit@gmail.com



## การแก้ไขล่าสุด (Fixes)

- แก้แอปพังทุกครั้งที่อัปโหลดไฟล์: `lief.parse(raw=...)` ใช้ไม่ได้แล้วใน lief เวอร์ชันใหม่
- ฟีเจอร์ที่แอปส่งเข้าโมเดลไม่ตรงกับที่ใช้เทรน (ลำดับ/ชนิดฟีเจอร์ผิด และไม่ได้ scale) ทำให้ผลทำนายสุ่ม
- แอปอ่านผลทำนายสลับคลาส (อ่านความน่าจะเป็นของ "benign" เป็น "malicious")
- โมเดลเดิมพึ่งฟีเจอร์ `appeared` และ `avclass` ซึ่งไม่มีในไฟล์ที่อัปโหลด (เหลือความแม่นยำ ~55%)
  จึงเทรนโมเดลใหม่ด้วยฟีเจอร์ที่คำนวณจากไฟล์ได้จริง บนข้อมูลมัลแวร์ชุดใหม่ EMBER2024

## Model

- Data: [EMBER2024](https://github.com/FutureComputing4AI/EMBER2024) — ไฟล์ Win32, Win64 และ .NET ที่ถูกพบบน VirusTotal ช่วง ก.ย. 2023 – ธ.ค. 2024
  - Train: 953,368 ไฟล์ (สุ่มอย่างสม่ำเสมอจาก 52 สัปดาห์แรก, benign/malicious อย่างละครึ่ง)
  - Test: 600,000 ไฟล์ จาก 12 สัปดาห์ถัดมา (ใหม่กว่าข้อมูลที่ใช้เทรนทั้งหมด)
- Features: 812 ฟีเจอร์จากตัวไฟล์ (byte histogram, byte-entropy histogram, strings, PE header, sections, imports, data directories) ตามนิยาม EMBER feature version 3 โดยใช้ `pefile`
- Neural network 512-256-128 (BatchNorm + Dropout)

| | Accuracy | ROC AUC |
|---|---|---|
| Test set (12 สัปดาห์หลัง train) | **97.15%** | **0.996** |

- Benign: precision 0.965 / recall 0.979 — Malicious: precision 0.979 / recall 0.964
- Challenge set (มัลแวร์ 6,315 ตัวที่ antivirus ~70 ตัวตรวจไม่เจอในตอนแรก): ตรวจเจอ 44.2%

### Files
- `streamlit_app.py` — เว็บแอป
- `features.py` — ดึงฟีเจอร์จากไฟล์ PE (ใช้ร่วมกันทั้งตอนเทรนและในแอป)
- `prepare_data.py` — แปลงไฟล์ zip ของ EMBER2024 เป็น `data/*.npz`
- `train_model.py` — เทรนโมเดล Neural Network
- `Virus_detect.keras`, `scaler.json` — โมเดลและค่า preprocessing

Retrain (ดาวน์โหลด `Win32_*.zip`, `Win64_*.zip`, `Dot_Net_*.zip`, `challenge.zip` จาก https://huggingface.co/datasets/joyce8/EMBER2024 ไว้ในโฟลเดอร์เดียวกัน):
```
pip install -r requirements.txt scikit-learn orjson
python prepare_data.py path/to/zip/folder
python train_model.py
```

Run locally:
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
