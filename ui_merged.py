# Dùng để deploy trực tiếp lên Streamlit Cloud

import streamlit as st
import uuid
import os
import json
from PIL import Image
from io import BytesIO
import Check_single as check_ransom_benign
import Check_single_ransome_mal as check_ransom_mal

# Các thư mục lưu file
UPLOAD_FOLDER = "analyze"
REPORT_FOLDER = "checkfile/reports"
PLOT_FOLDER = "checkfile/plots"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Giao diện
st.title("🛡️ Ransomware Analyzer")
analysis_type = st.selectbox("Chọn loại phân tích:", ["Ransomware and Benign", "Ransomware and Malware"])
input_type = st.selectbox("Chọn kiểu file đầu vào:", ["execute", "report", "attribute"])

# Tải file
if input_type == "execute":
    uploaded_file = st.file_uploader("Tải lên file .exe")
else:
    uploaded_file = st.file_uploader("Tải lên file JSON", type=["json"])

if uploaded_file and st.button("Phân tích"):
    # Tạo filename ngẫu nhiên
    filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Lưu file upload
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Chọn module
    module = check_ransom_benign if analysis_type == "Ransomware and Benign" else check_ransom_mal

    try:
        if input_type == "execute":
            task_id = module.submit_sample(filepath)
            module.wait_for_report(task_id)

            report_path = os.path.join(REPORT_FOLDER, f"report_{filename}.json")
            module.download_report(task_id, report_path)

            attr_path = os.path.join(REPORT_FOLDER, f"attributes_{filename}.json")
            module.extract_fields(report_path, attr_path)

        elif input_type == "report":
            attr_path = os.path.join(REPORT_FOLDER, f"attributes_{filename}.json")
            module.extract_fields(filepath, attr_path)

        elif input_type == "attribute":
            attr_path = filepath

        else:
            st.error("❌ Kiểu đầu vào không hợp lệ")
            st.stop()

        # Phân tích model
        result = module.check_and_explain(attr_path)
        if result is None or result[0] == "Invalid":
            st.error("❌ Dữ liệu đầu vào không hợp lệ")
            st.stop()

        label, confidence, lime_result = result

        # Vẽ LIME
        plot_filename = f"plot_{uuid.uuid4()}.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)
        module.plot_lime_top_5_5_to_file(lime_result, filename, label, confidence, save_path=plot_path)

        # Hiển thị kết quả
        st.success("✅ Phân tích thành công")
        st.markdown(f"### 🔖 Nhãn: **{label}**")
        st.markdown(f"**Độ tin cậy:** {confidence:.4f}")

        st.markdown("### 🔍 Top LIME features:")
        for tok, w in lime_result:
            st.markdown(f"- `{tok}`: **{w:.4f}**")

        # Hiển thị ảnh
        with open(plot_path, "rb") as f:
            img = Image.open(BytesIO(f.read()))
            st.image(img, caption="Biểu đồ LIME")

    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
