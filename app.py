import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Cấu hình giao diện
st.set_page_config(
    page_title="Phân Nhóm Nguyên Liệu Thực Phẩm", 
    page_icon="🥗", 
    layout="wide"
)

st.title("🥗 Hệ Thống Quét & Phân Nhóm Nguyên Liệu (Thịt, Rau, Trứng, Cá...)")
st.markdown("Ứng dụng tự động nhận diện và gom nhóm nguyên liệu thô vào các nhóm dinh dưỡng cơ bản.")

# Thanh sidebar cấu hình
confidence_threshold = st.sidebar.slider(
    "Độ tự tin tối thiểu (Confidence)", 
    min_value=0.1, max_value=0.9, value=0.3, step=0.05,
    help="Điều chỉnh mức độ nhạy của AI khi quét."
)

# Tải mô hình YOLOv8 tiêu chuẩn
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Lỗi tải mô hình: {e}")

# Phân loại tự động vào các nhóm nguyên liệu chung
CATEGORY_MAPPING = {
    "🥩 Thịt (Meat/Chicken/Pork)": {
        "keywords": ["meat", "beef", "pork", "chicken", "bird", "animal"],
        "color": "#ff4b4b",
        "avg_calo": 250
    },
    "🐟 Cá & Hải sản (Fish/Seafood)": {
        "keywords": ["fish", "shrimp", "crab", "seafood", "lobster"],
        "color": "#1c83e1",
        "avg_calo": 130
    },
    "🥦 Rau củ (Vegetables)": {
        "keywords": ["carrot", "broccoli", "vegetable", "plant", "salad", "cabbage", "potato", "tomato"],
        "color": "#28a745",
        "avg_calo": 40
    },
    "🍎 Hoa quả (Fruits)": {
        "keywords": ["apple", "banana", "orange", "fruit", "lemon"],
        "color": "#ffc107",
        "avg_calo": 60
    },
    "🥚 Trứng & Khác (Eggs)": {
        "keywords": ["egg"],
        "color": "#fd7e14",
        "avg_calo": 155
    }
}

def categorize_item(class_name):
    class_lower = class_name.lower()
    for cat_name, info in CATEGORY_MAPPING.items():
        if any(kw in class_lower for kw in info["keywords"]):
            return cat_name, info["color"], info["avg_calo"]
    return "📌 Nguyên liệu / Vật thể khác", "#6c757d", 100

# Khu vực tải ảnh
uploaded_file = st.file_uploader("📂 Tải lên ảnh nguyên liệu (Thịt, rau, củ, quả, trứng...)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("🖼️ Ảnh gốc")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 Quét Phân Nhóm Nguyên Liệu", type="primary"):
        with st.spinner("AI đang quét và phân nhóm nguyên liệu..."):
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
                st.subheader("📊 Bảng Phân Nhóm Nguyên Liệu & Calo Ước Tính")
                
                grouped_data = {}
                total_calo = 0
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    conf = float(box.conf[0]) * 100
                    
                    category, color, calo = categorize_item(class_name)
                    
                    if category not in grouped_data:
                        grouped_data[category] = []
                    grouped_data[category].append((class_name, conf, calo))
                    total_calo += calo

                for category, items in grouped_data.items():
                    st.markdown(f"### {category}")
                    for name, conf, calo in items:
                        st.markdown(
                            f"""
                            <div style="padding: 8px; border-radius: 5px; background-color: #f8f9fa; border-left: 4px solid {color}; margin-bottom: 6px;">
                                🔹 Nhận diện: <b>{name.upper()}</b> | Độ tin cậy: <code>{conf:.1f}%</code> | Calo ước tính: <code>~{calo} kcal</code>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
                st.markdown("---")
                c1, c2 = st.columns(2)
                c1.metric("Tổng số mục phát hiện", f"{len(boxes)} mục")
                c2.metric("Tổng năng lượng ước tính", f"~{total_calo} kcal")
            else:
                st.warning("⚠️ Không tìm thấy nguyên liệu nào đạt độ tin cậy trong ảnh.")
else:
    st.info("👆 Vui lòng tải một hình ảnh nguyên liệu lên để bắt đầu.")