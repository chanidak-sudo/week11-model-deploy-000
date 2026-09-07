# app.py
# -*- coding: utf-8 -*-
# ==================================================================
#  โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray (Streamlit)
# ------------------------------------------------------------------
#  โมเดล (*.pkcls) เป็นโมเดลของ "Orange Data Mining"
#  ที่เทรนบน "image embedding" ขนาด 2048 มิติ ซึ่งได้จากโมเดล
#  Inception v3 (วิดเจ็ต Image Embedding ของ Orange)
#
#  ลำดับการทำงาน:
#    ผู้ใช้อัปโหลดภาพ X-ray  ->  ฝังภาพเป็นเวกเตอร์ 2048 ค่า
#    ->  ส่งเข้าโมเดล Orange ที่เลือก  ->  ทำนายเป็น covid / normal / pneumonia
# ==================================================================

import os
# ตั้งค่าให้ Qt ทำงานแบบไม่ต้องมีจอ (จำเป็นเมื่อรันบนเซิร์ฟเวอร์/คลาวด์)
# ต้องตั้งค่า "ก่อน" import ส่วนที่เกี่ยวกับ Orange image embedder
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import glob                  # ใช้ค้นหาไฟล์ตามรูปแบบ *.pkcls
import tempfile              # ใช้สร้างไฟล์ภาพชั่วคราวสำหรับส่งให้ตัวฝังภาพ
import joblib                # ใช้โหลดโมเดลที่บันทึกไว้
import numpy as np           # ใช้จัดการเวกเตอร์ตัวเลข
import streamlit as st       # ไลบรารีสร้างเว็บแอป

# --- ส่วนของ Orange ---
from Orange.data import Table, Domain   # ใช้สร้างตารางข้อมูลรูปแบบ Orange
# ตัวฝังภาพ (image embedder) ของ Orange ที่ให้เวกเตอร์ 2048 ค่าแบบ Inception v3
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

# ------------------------------------------------------------------
#  ค่าคงที่ของแอป
# ------------------------------------------------------------------
MODELS_DIR = "Models"          # โฟลเดอร์ที่เก็บไฟล์ *.pkcls (วางข้างไฟล์ app.py นี้)
EMBEDDER_MODEL = "inception-v3"  # ตัวฝังภาพ ต้องตรงกับที่ใช้ตอนเทรนใน Orange

# แปลงชื่อคลาส (ภาษาอังกฤษของโมเดล) เป็นภาษาไทยเพื่อให้อ่านง่าย
LABEL_TH = {
    "covid": "ติดเชื้อโควิด-19",
    "normal": "ปกติ",
    "pneumonia": "ปอดอักเสบ (Pneumonia)",
}


# ------------------------------------------------------------------
#  ฟังก์ชันช่วยเหลือ
# ------------------------------------------------------------------
def list_model_files(models_dir: str):
    """คืนรายการ path ของไฟล์ *.pkcls ทั้งหมดในโฟลเดอร์ที่กำหนด"""
    return sorted(glob.glob(os.path.join(models_dir, "*.pkcls")))


@st.cache_resource   # แคชโมเดลไว้ ไม่ต้องโหลดใหม่ทุกครั้ง
def load_model(model_path: str):
    """โหลดโมเดล Orange จากไฟล์ .pkcls ด้วย joblib"""
    return joblib.load(model_path)


@st.cache_resource   # แคชตัวฝังภาพไว้ใช้ซ้ำ (สร้างครั้งเดียว)
def get_embedder():
    """สร้างตัวฝังภาพ Inception v3 (ต้องเชื่อมต่ออินเทอร์เน็ตเพื่อเรียกเซิร์ฟเวอร์ฝังภาพของ Orange)"""
    return ImageEmbedder(model=EMBEDDER_MODEL)


def embed_image(image_path: str):
    """
    แปลงไฟล์ภาพ 1 รูป ให้เป็นเวกเตอร์ตัวเลข 2048 ค่า (image embedding)
    คืนค่าเป็น numpy array รูปร่าง (1, 2048) หรือ None ถ้าฝังภาพไม่สำเร็จ
    """
    embedder = get_embedder()
    # ImageEmbedder รับ "ลิสต์ของ path" และคืน "ลิสต์ของเวกเตอร์"
    embeddings = embedder([image_path])
    vec = embeddings[0]                       # เอาผลของภาพแรก
    if vec is None or len(vec) == 0:          # ภาพนี้ถูกข้าม (embedding ไม่สำเร็จ)
        return None
    return np.asarray(vec, dtype=float).reshape(1, -1)


def predict(model, feature_vector: np.ndarray):
    """
    นำเวกเตอร์ฟีเจอร์ (1, 2048) เข้าโมเดล Orange แล้วคืน (ชื่อคลาสที่ทำนาย, ตารางความน่าจะเป็น)
    """
    # ใช้ 'original_domain' ของโมเดล (โดเมนอินพุตดิบ n0..n2047)
    # เพื่อให้ Orange แปลง/พรีโพรเซสภายในได้ถูกต้อง (สำคัญมากสำหรับ SVM/NN)
    input_domain = Domain(model.original_domain.attributes)
    table = Table.from_numpy(input_domain, feature_vector)

    # ทำนายคลาส (ได้เป็นดัชนีของคลาส) แล้วแปลงเป็นชื่อคลาส
    class_index = int(model(table)[0])
    class_name = model.domain.class_var.values[class_index]

    # ทำนายความน่าจะเป็นของแต่ละคลาส (ถ้าโมเดลรองรับ)
    probs = None
    try:
        from Orange.classification import Model as OrangeModel
        proba_row = model(table, OrangeModel.Probs)[0]     # เวกเตอร์ความน่าจะเป็น
        probs = list(zip(model.domain.class_var.values, proba_row))
    except Exception:
        probs = None

    return class_name, probs


# ==================================================================
#  ส่วนติดต่อผู้ใช้ (UI)
# ==================================================================

# --- หัวข้อแอป ---
st.title("โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray")

# --- แถบด้านข้าง: เลือกโมเดล ---
st.sidebar.header("เลือกโมเดล")
model_files = list_model_files(MODELS_DIR)

if not model_files:
    # ไม่พบไฟล์โมเดล -> แจ้งเตือนแล้วหยุด
    st.error(
        f"ไม่พบไฟล์ *.pkcls ในโฟลเดอร์ '{MODELS_DIR}' "
        f"กรุณานำไฟล์โมเดล (เช่น W10-SVM-Model.pkcls) มาไว้ในโฟลเดอร์นี้ก่อน"
    )
    st.stop()

# ให้ผู้ใช้เลือกไฟล์โมเดลเอง (แสดงเฉพาะชื่อไฟล์)
selected_model_path = st.sidebar.selectbox(
    "ไฟล์โมเดล (*.pkcls)",
    options=model_files,
    format_func=lambda p: os.path.basename(p),
)

# โหลดโมเดลที่เลือก
model = load_model(selected_model_path)
st.sidebar.success(f"โหลดโมเดลสำเร็จ: {os.path.basename(selected_model_path)}")
# แสดงชื่อคลาสที่โมเดลจำแนกได้
st.sidebar.caption("คลาสที่ทำนายได้: " + ", ".join(model.domain.class_var.values))

# --- ส่วนอัปโหลดภาพ ---
st.subheader("อัปโหลดภาพ X-ray")
uploaded_file = st.file_uploader(
    "เลือกไฟล์ภาพ (.png, .jpg, .jpeg)",
    type=["png", "jpg", "jpeg"],
)

# แสดงตัวอย่างภาพที่อัปโหลด
if uploaded_file is not None:
    st.image(uploaded_file, caption="ภาพที่อัปโหลด", use_container_width=True)

# --- ปุ่มทำนายผล ---
if st.button("ทำนายผล"):
    if uploaded_file is None:
        st.warning("กรุณาอัปโหลดภาพ X-ray ก่อนกดทำนายผล")
    else:
        # 1) บันทึกภาพลงไฟล์ชั่วคราว (ตัวฝังภาพต้องอ่านจาก path ของไฟล์)
        suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            # 2) ฝังภาพเป็นเวกเตอร์ 2048 ค่า
            with st.spinner("กำลังประมวลผลภาพ (ฝังภาพด้วย Inception v3)..."):
                feature_vector = embed_image(tmp_path)

            if feature_vector is None:
                st.error("ไม่สามารถฝังภาพนี้ได้ กรุณาลองภาพอื่น")
            else:
                # 3) ส่งเวกเตอร์เข้าโมเดลเพื่อทำนาย
                class_name, probs = predict(model, feature_vector)

                # 4) แสดงผลลัพธ์ให้อ่านง่าย
                label_th = LABEL_TH.get(class_name, class_name)
                st.success(f"ผลการทำนาย: {label_th}  (คลาส: {class_name})")

                # แสดงตารางความน่าจะเป็นของแต่ละคลาส (ถ้ามี)
                if probs is not None:
                    import pandas as pd
                    proba_df = pd.DataFrame(
                        [
                            {
                                "คลาส": LABEL_TH.get(name, name),
                                "ความน่าจะเป็น": f"{p*100:.2f}%",
                            }
                            for name, p in probs
                        ]
                    )
                    st.write("ความน่าจะเป็นของแต่ละคลาส:")
                    st.dataframe(proba_df, hide_index=True, use_container_width=True)

        except Exception as e:
            # ข้อผิดพลาดที่พบบ่อยคือ ต่ออินเทอร์เน็ตไม่ได้ (เซิร์ฟเวอร์ฝังภาพของ Orange)
            st.error(f"เกิดข้อผิดพลาด: {e}")
            st.info(
                "การฝังภาพต้องเชื่อมต่ออินเทอร์เน็ต (เรียกเซิร์ฟเวอร์ของ Orange) "
                "กรุณาตรวจสอบการเชื่อมต่อ แล้วลองใหม่อีกครั้ง"
            )
        finally:
            # ลบไฟล์ชั่วคราวทิ้ง
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
