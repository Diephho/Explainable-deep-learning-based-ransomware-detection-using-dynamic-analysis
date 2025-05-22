import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("🛡️ Malware Analyzer")

analysis_type = st.selectbox("Chọn loại phân tích:", ["Ransomware and Benign", "Ransomware and Malware"])
input_type = st.selectbox("Chọn kiểu file đầu vào:", ["execute", "report", "attribute"])

if input_type == "execute":
    uploaded_file = st.file_uploader("Tải lên file")
else:
    uploaded_file = st.file_uploader("Tải lên file", type=["json"])

if uploaded_file and st.button("Phân tích"):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    data = {
        "analysis_type": analysis_type,
        "input_type": input_type
    }

    try:
        res = requests.post("http://localhost:5000/analyze", files=files, data=data)
        res.raise_for_status()
        result = res.json()

        if "error" in result:
            st.error(f"❌ Lỗi: {result['error']}")
        else:
            st.success("✅ Phân tích thành công!")

            st.markdown(f"### 🔖 Nhãn dự đoán: **{result['label']}**")
            st.markdown(f"**Độ tin cậy:** {result['confidence']:.2f}")

            st.markdown("### 🔍 Top LIME features:")
            for token, weight in result["lime_features"]:
                st.markdown(f"- `{token}`: **{weight:.4f}**")

            # Hiển thị plot
            plot_resp = requests.get(f"http://localhost:5000{result['plot_url']}")
            img = Image.open(BytesIO(plot_resp.content))
            st.image(img, caption="Biểu đồ LIME")

    except Exception as e:
        st.error(f"❌ Lỗi phân tích: {e}")
