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
st.markdown("Hệ thống thông minh giúp phân nhóm nguyên liệu (Thịt, Cá, Rau, Quả, Trứng) và ước tính dinh dưỡng tự động.")

# Thanh sidebar cấu hình nâng cao
st.sidebar.header("⚙️ Tùy Chỉnh AI")
confidence_threshold = st.sidebar.slider(
    "Độ tự tin tối thiểu (Confidence)", 
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Tăng mức này lên nếu mô hình hay nhận diện nhầm lẫn (ví dụ: cá thành thịt bò)."
)

# Tải mô hình YOLOv8
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Lỗi khi tải mô hình 'best.pt': {e}. Hãy đảm bảo file mô hình nằm cùng thư mục với `app.py` trên GitHub!")

# Phân loại nguyên liệu vào các nhóm cụ thể kèm theo lượng calo ước tính trung bình (trên 100g)
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
    
    # Nút thực thi
    if st.button("🚀 Quét và Phân Loại Nguyên Liệu", type="primary"):
        with st.spinner("AI đang phân tích từng chi tiết nguyên liệu..."):
            # Chạy model với ngưỡng tin cậy tùy chỉnh từ sidebar
            results = model(image, conf=confidence_threshold)
            res = results[0]
            
            # Vẽ khung kết quả lên ảnh
            res_plotted = res.plot()
            res_plotted_rgb = Image.fromarray(res_plotted[..., ::-1])
            
            with col2:
                st.subheader("🎯 Kết quả nhận diện")
                st.image(res_plotted_rgb, use_container_width=True)
            
            boxes = res.boxes
            if len(boxes) > 0:
                st.markdown("---")
                st.subheader("📊 Bảng Thống Kê & Phân Nhóm Nguyên Liệu")
                
                # Gom nhóm dữ liệu
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

                # Hiển thị kết quả chia theo từng tab hoặc từng card nhóm màu sắc
                cols = st.columns(len(grouped_data))
                idx = 0
                
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
                
                # Tổng hợp dinh dưỡng nhanh
                st.markdown("---")
                m1, m2 = st.columns(2)
                m1.metric(label="Tổng số nguyên liệu phát hiện", value=f"{len(boxes)} món")
                m2.metric(label="Tổng năng lượng ước tính (Mẫu chuẩn)", value=f"~{total_estimated_calo} kcal")
                
                st.info("💡 **Mẹo khắc phục lỗi nhận diện nhầm (Cá thành Thịt bò):** Nếu bạn thấy nhãn bị gán sai nhóm, hãy kiểm tra lại tập dữ liệu (dataset) lúc train hoặc tăng thanh trượt **Độ tự tin** ở thanh menu bên trái lên mức `0.6` hoặc `0.7` để lọc bớt các kết quả nhiễu.")
            else:
                st.warning("⚠️ Không tìm thấy nguyên liệu nào đạt độ tin cậy theo cấu hình hiện tại. Hãy thử hạ thấp độ tự tin ở sidebar hoặc chụp ảnh rõ nét hơn!")
else:
    st.info("👆 Vui lòng tải một hình ảnh thực phẩm lên để bắt đầu hệ thống phân loại tự động.")