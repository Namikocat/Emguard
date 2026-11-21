ยินดีต้อนรับทุกคนสู่ โปรเจคEmguard
โปรเจคนี้คือโปรเจคในการ Detect ไวรัสคอมพิวเตอร์โดยเพียงแค่อัปไฟล์ 

Data set of this project
Ember Data set Frome this link
https://ember.elastic.co/ember_dataset_2018_2.tar.bz2

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
  จึงเทรนโมเดลใหม่ด้วยฟีเจอร์ที่คำนวณจากไฟล์ได้จริง ได้ความแม่นยำบน test set **~90%**

### Files
- `streamlit_app.py` — เว็บแอป
- `features.py` — ดึงฟีเจอร์จากไฟล์ PE ตามนิยามของ EMBER
- `train_model.py` — เทรนโมเดลใหม่ (`python train_model.py ember2018/train_features_1.jsonl`)
- `Virus_detect.keras`, `scaler.json` — โมเดลและค่า preprocessing

Run locally:
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
