import base64
import hashlib
import io
import json
import urllib.request
import streamlit as st
from PIL import Image

# Cấu hình giao diện
st.set_page_config(
    page_title="NutriPath AI - Nhận Diện Thực Phẩm Thực Tế", 
    page_icon="🥗", 
    layout="wide"
)

st.title("🥗 NutriPath AI - Quét Đồ Ăn & Phân Tích Dinh Dưỡng")
st.markdown("Hệ thống AI thông minh giúp nhận diện chính xác món ăn thực tế, ước tính calo và cảnh báo theo bệnh lý.")

# --- SIDEBAR: CẤU HÌNH API & HỒ SƠ BỆNH LÝ ---
with st.sidebar:
    st.header("🔑 Cấu Hình AI Nhận Diện")
    saved_key = ""
    try:
        saved_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        saved_key = ""

    api_key = st.text_input(
        "Nhập Gemini API Key:",
        value=saved_key,
        type="password",
        help="Lấy mã miễn phí tại https://aistudio.google.com/app/apikey"
    )
    
    if not api_key.strip():
        st.warning("⚠️ Vui lòng nhập Gemini API Key để kích hoạt khả năng quét ảnh thực tế!")

    st.markdown("---")
    st.header("🏥 Hồ Sơ Bệnh Lý Cá Nhân")
    st.caption("Tích chọn để nhận cảnh báo dinh dưỡng:")
    cond_diabetes = st.checkbox("Đái tháo đường (Tiểu đường)")
    cond_hypertension = st.checkbox("Tăng huyết áp (Cao huyết áp)")
    cond_gout = st.checkbox("Bệnh Gout (Tăng Acid Uric)")
    cond_lipid = st.checkbox("Rối loạn mỡ máu / Tim mạch")

# --- HÀM GỌI GEMINI API NHẬN DIỆN ẢNH THỰC TẾ ---
def scan_food_with_gemini(image_pil: Image.Image, key: str) -> dict:
    buffered = io.BytesIO()
    image_pil.convert("RGB").save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key.strip()}"
    
    prompt = """
    Quan sát bức ảnh này và trả về kết quả đúng định dạng JSON thuần túy (không dùng markdown):
    {
      "is_food": true/false,
      "dish_name": "Tên món ăn hoặc nguyên liệu chính bằng tiếng Việt (Ví dụ: Thịt xiên nướng, Phở bò, Đĩa rau muống xào... Nếu không phải đồ ăn để trống '')",
      "categories": [
        {
          "group_name": "Nhóm nguyên liệu (Chọn 1 trong các nhóm: '🥩 Thịt & Đạm', '🐟 Cá & Hải sản', '🥦 Rau củ quả', '🥚 Trứng & Sữa', '🍚 Tinh bột / Món chính', '🍎 Trái cây')",
          "items": ["chi tiết nguyên liệu phát hiện"],
          "estimated_calo": số_calo_ước_tính
        }
      ],
      "total_calo": tổng_calo_ước_tính,
      "sodium_mg": lượng_natri_mg_ước_tính,
      "carbs_g": lượng_carbs_g_ước_tính,
      "sugar_g": lượng_đường_g_ước_tính,
      "protein_g": lượng_đạm_g_ước_tính,
      "fat_g": lượng_chất_béo_g_ước_tính,
      "advice": "Lời khuyên dinh dưỡng ngắn gọn"
    }
    """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)

# --- QUẢN LÝ TRẠNG THÁI XÓA KẾT QUẢ CŨ KHI ĐỔI ẢNH ---
def clear_state():
    if "scan_result" in st.session_state:
        st.session_state.scan_result = None

if "last_hash" not in st.session_state:
    st.session_state.last_hash = None
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

# --- KHU VỰC TẢI ẢNH HOẶC CHỤP CAMERA ---
tab1, tab2 = st.tabs(["📂 Tải ảnh từ thiết bị", "📸 Chụp ảnh từ Camera"])
with tab1:
    uploaded_file = st.file_uploader("Chọn hình ảnh món ăn hoặc nguyên liệu (JPG, PNG)...", type=["jpg", "jpeg", "png", "webp"])
with tab2:
    camera_file = st.camera_input("Chụp ảnh thực tế")

current_file = camera_file if camera_file is not None else uploaded_file

if current_file is not None:
    file_bytes = current_file.getvalue()
    img_hash = hashlib.md5(file_bytes).hexdigest()

    # Tự động xóa kết quả cũ ngay khi đổi ảnh mới
    if st.session_state.last_hash != img_hash:
        st.session_state.last_hash = img_hash
        clear_state()

    image = Image.open(current_file)
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("🖼️ Ảnh thực tế")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Kết quả AI phân tích")
        if st.button("🚀 Quét Đồ Ăn Ngay Lập Tức", type="primary", use_container_width=True):
            clear_state()
            if not api_key.strip():
                st.error("❌ Vui lòng nhập Gemini API Key ở thanh sidebar bên trái để sử dụng tính năng quét ảnh!")
            else:
                with st.spinner("AI đang phân tích bức ảnh của bạn..."):
                    try:
                        result = scan_food_with_gemini(image, api_key)
                        st.session_state.scan_result = result
                    except Exception as e:
                        st.error(f"Lỗi kết nối khi quét ảnh: {e}")

        # Hiển thị kết quả quét
        res = st.session_state.get("scan_result")
        if res is not None:
            if not res.get("is_food", True):
                st.warning("⚠️ AI nhận thấy đây có vẻ không phải là hình ảnh thực phẩm. Hãy thử chụp lại rõ hơn nhé!")
            else:
                st.success(f"🎯 **Tên nhận diện:** {res.get('dish_name', 'Món ăn thực tế')}")
                
                st.markdown("---")
                st.markdown("#### 📊 Phân Nhóm Nguyên Liệu")
                for cat in res.get("categories", []):
                    st.markdown(
                        f"""
                        <div style="padding: 8px; border-radius: 6px; background-color: #f8f9fa; border-left: 5px solid #28a745; margin-bottom: 6px;">
                            <b>{cat.get('group_name')}</b><br>
                            🔹 Chi tiết: <code>{', '.join(cat.get('items', []))}</code><br>
                            🔥 Calo: <code>~{cat.get('estimated_calo')} kcal</code>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("🔥 Tổng Calo", f"~{res.get('total_calo', 350)} kcal")
                m2.metric("🧂 Natri", f"~{res.get('sodium_mg', 500)} mg")
                m3.metric("🥩 Đạm", f"~{res.get('protein_g', 20)} g")

                if res.get("advice"):
                    st.info(f"💡 **Lời khuyên:** {res.get('advice')}")

                # --- PHÂN TÍCH BỆNH LÝ THEO LỰA CHỌN ---
                st.markdown("---")
                st.subheader("🏥 Phân Tích Theo Bệnh Lý")
                if not any([cond_diabetes, cond_hypertension, cond_gout, cond_lipid]):
                    st.caption("👉 Hãy tích chọn bệnh lý ở thanh bên trái (Sidebar) để xem cảnh báo chi tiết.")
                else:
                    if cond_diabetes and (res.get("carbs_g", 0) >= 45 or res.get("sugar_g", 0) >= 8):
                        st.error("🔴 **Đái tháo đường:** Món này có lượng Carbs hoặc Đường tương đối cao. Nên cân nhắc giảm khẩu phần.")
                    if cond_hypertension and (res.get("sodium_mg", 0) >= 700):
                        st.error("🟠 **Tăng huyết áp:** Lượng Natri (muối) khá cao. Hạn chế húp nước dùng hoặc chấm thêm gia vị.")
                    if cond_gout and (res.get("protein_g", 0) >= 25):
                        st.warning("🟡 **Bệnh Gout:** Lượng đạm cao, cần lưu ý hạn chế thịt đỏ hoặc nước hầm xương.")
                    if cond_lipid and (res.get("fat_g", 0) >= 18):
                        st.warning("🟡 **Mỡ máu:** Lượng chất béo khá cao, nên hạn chế phần mỡ/da động vật.")
                    
                    st.caption("⚕️ *Kết quả chỉ mang tính tham khảo dinh dưỡng.*")
else:
    clear_state()
    st.info("👆 Vui lòng tải lên hoặc chụp một bức ảnh thực tế để bắt đầu quét.")