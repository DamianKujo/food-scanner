import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="Hệ Thống Phân Loại & Ước Tính Calo Thực Phẩm", 
    page_icon="🥗", 
    layout="wide"
)

st.title("🥗 Ứng Dụng AI Nhận Diện & Phân Loại Nguyên Liệu Thức Ăn")
st.markdown("Hệ thống thông minh giúp phân nhóm nguyên liệu và ước tính dinh dưỡng tự động.")

# Thanh sidebar cấu hình nâng cao
st.sidebar.header("⚙️ Tùy Chỉnh AI")
confidence_threshold = st.sidebar.slider(
    "Độ tự tin tối thiểu (Confidence)", 
    min_value=0.1, max_value=0.9, value=0.3, step=0.05,
    help="Tăng mức này lên nếu muốn lọc bỏ kết quả nhiễu."
)

# Sử dụng model chuẩn yolov8n.pt có sẵn (không cần file best.pt nữa)
@st.cache_resource
def load_model():
    # Tự động tải model cơ bản của YOLOv8
    model = YOLO("yolov8n.pt")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Lỗi khi tải mô hình: {e}")

# Khu vực chính tải ảnh
uploaded_file = st.file_uploader("📂 Tải lên hình ảnh nguyên liệu hoặc món ăn (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("🖼️ Ảnh gốc")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 Quét và Phân Loại Nguyên Liệu", type="primary"):
        with st.spinner("AI đang phân tích hình ảnh..."):
            results = model(image, conf=confidence_threshold)
            res = results[0]
            
            res_plotted = res.plot()
            res_plotted_rgb = Image.fromarray(res_plotted[..., ::-1])
            
            with col2:
                st.subheader("🎯 Kết quả nhận diện")
                st.image(res_plotted_rgb, use_container_width=True)
            
            boxes = res.boxes
            if len(boxes) > 0:
                st.markdown("---")
                st.subheader("📊 Danh Sách Vật Thể Nhận Diện Được")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0]) * 100
                    st.markdown(f"- **{class_name.upper()}** (Độ tin cậy: `{confidence:.1f}%`)")
            else:
                st.warning("⚠️ Không tìm thấy vật thể nào đạt độ tin cậy theo cấu hình hiện tại.")
else:
    st.info("👆 Vui lòng tải một hình ảnh lên để bắt đầu.")