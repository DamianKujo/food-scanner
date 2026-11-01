import os
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import gdown

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="Hệ Thống Phân Loại & Ước Tính Calo Thực Phẩm", 
    page_icon="🥗", 
    layout="wide"
)

st.title("🥗 Ứng Dụng AI Nhận Diện & Phân Loại Nguyên Liệu Thức Ăn")
st.markdown("Hệ thống thông minh giúp phân nhóm nguyên liệu (Thịt, Cá, Rau, Quả, Trứng) và ước tính dinh dưỡng tự động.")

# Thanh sidebar cấu hình nâng cao
st.sidebar.header("⚙️ Tùy Chỉnh AI")
confidence_threshold = st.sidebar.slider(
    "Độ tự tin tối thiểu (Confidence)", 
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Tăng mức này lên nếu mô hình hay nhận diện nhầm lẫn."
)

# Tự động tải file best.pt từ Google Drive nếu chưa có trên server
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        # ⚠️ BẠN HÃY THAY LINK GOOGLE DRIVE CỦA FILE best.pt VÀO ĐÂY (Link phải ở chế độ công khai)
        # Ví dụ định dạng link: 'https://drive.google.com/uc?id=MÃ_FILE_GOOGLE_DRIVE'
        file_id = "DIEN_FILE_ID_GOOGLE_DRIVE_CUA_BAN_VAO_DAY"
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Không thể tự động tải mô hình từ Google Drive: {e}")
            
    model = YOLO(model_path)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Lỗi khi tải mô hình: {e}. Vui lòng kiểm tra lại link Google Drive trong code `app.py`!")

# Phân loại nguyên liệu vào các nhóm cụ thể kèm theo lượng calo ước tính trung bình
CATEGORY_MAPPING = {
    "🥩 Thịt (Meat/Beef/Pork/Chicken)": {
        "keywords": ["meat", "beef", "pork", "chicken", "thit", "bo", "heo", "ga"],
        "color": "#ff4b4b",
        "avg_calo": 250
    },
    "🐟 Cá & Hải sản (Fish/Seafood)": {
        "keywords": ["fish", "seafood", "shrimp", "crab", "ca", "tom", "cua", "muc"],
        "color": "#1c83e1",
        "avg_calo": 130
    },
    "🥦 Rau củ (Vegetables)": {
        "keywords": ["vegetable", "veg", "cabbage", "carrot", "rau", "cu", "cai", "ca_rot", "broccoli"],
        "color": "#28a745",
        "avg_calo": 40
    },
    "🍎 Hoa quả (Fruits)": {
        "keywords": ["fruit", "apple", "banana", "orange", "hoa_qua", "trai_cay", "tao", "chuoi", "xoai"],
        "color": "#ffc107",
        "avg_calo": 60
    },
    "🥚 Trứng (Eggs)": {
        "keywords": ["egg", "trung"],
        "color": "#fd7e14",
        "avg_calo": 155
    }
}

def categorize_item(class_name):
    class_lower = class_name.lower()
    for cat_name, info in CATEGORY_MAPPING.items():
        if any(kw in class_lower for kw in info["keywords"]):
            return cat_name, info["color"], info["avg_calo"]
    return "📌 Nguyên liệu khác", "#6c757d", 100

# Khu vực chính tải ảnh
uploaded_file = st.file_uploader("📂 Tải lên hình ảnh nguyên liệu hoặc món ăn (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("🖼️ Ảnh gốc")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 Quét và Phân Loại Nguyên Liệu", type="primary"):
        with st.spinner("AI đang phân tích từng chi tiết nguyên liệu..."):
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
                st.subheader("📊 Bảng Thống Kê & Phân Nhóm Nguyên Liệu")
                
                grouped_data = {}
                total_estimated_calo = 0
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0]) * 100
                    
                    category, color, calo = categorize_item(class_name)
                    
                    if category not in grouped_data:
                        grouped_data[category] = []
                    grouped_data[category].append((class_name, confidence, calo))
                    total_estimated_calo += calo

                for category, items in grouped_data.items():
                    with st.container():
                        st.markdown(f"### {category}")
                        for name, conf, calo in items:
                            st.markdown(
                                f"""
                                <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa; border-left: 5px solid #ff4b4b; margin-bottom: 8px;">
                                    <b>{name.upper()}</b><br>
                                    🔹 Độ tin cậy: <code>{conf:.1f}%</code><br>
                                    🔥 Calo ước tính: <code>~{calo} kcal</code>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                
                st.markdown("---")
                m1, m2 = st.columns(2)
                m1.metric(label="Tổng số nguyên liệu phát hiện", value=f"{len(boxes)} món")
                m2.metric(label="Tổng năng lượng ước tính", value=f"~{total_estimated_calo} kcal")
            else:
                st.warning("⚠️ Không tìm thấy nguyên liệu nào đạt độ tin cậy theo cấu hình hiện tại.")
else:
    st.info("👆 Vui lòng tải một hình ảnh thực phẩm lên để bắt đầu hệ thống phân loại tự động.")