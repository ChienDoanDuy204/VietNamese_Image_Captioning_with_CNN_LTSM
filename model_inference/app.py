"""
Image Captioning App - Streamlit UI
-------------------------------------
Giao diện web cho hệ thống sinh caption từ ảnh (image -> text).
Hỗ trợ upload file ảnh hoặc chụp trực tiếp từ camera.

Chạy thử:
    streamlit run app.py
"""

import time
from io import BytesIO

import streamlit as st
from PIL import Image

############################################### PREPARE MODEL #######################################################
from pathlib import Path
import sys
import streamlit as st

ROOT_DIR = Path.cwd().parent
MODEL_DIR = ROOT_DIR/'model'
CONFIG_DIR = ROOT_DIR/'configuration'
SOURCE_DIR = ROOT_DIR/'source'
# Thêm các thư mục vào không gian tìm kiếm 

sys.path.extend([str(ROOT_DIR), str(MODEL_DIR), str(CONFIG_DIR), str(SOURCE_DIR)])

from model.build_model import *
from source.preprocessing.preprocessing_img import *
from source.vocabulary.vocab import *
# Path save vocab and the parameter save mean and std of image

PATH_SAVE_VOCAB = CONFIG_DIR/'vocab.json'
PATH_SAVE_MEAN_STD = CONFIG_DIR/'mean_std_img.json'
PATH_WEIGHT_MODEL = ROOT_DIR/'model_weight.pth'

vocab = BuiltVocabFromIterator()
vocab.load(path = str(PATH_SAVE_VOCAB))

computer_mean_std = ComputeMeanStd()
mean, std = computer_mean_std.load_mean_std(path = str(PATH_SAVE_MEAN_STD))

def prepare_model_and_generate_caption(img):
    processor_img_upload = process_Image_upload(mean=mean, std = std, img_size=(224, 224))
    img = processor_img_upload(img=img)
    # Load weight
    state_dict = torch.load(str(PATH_WEIGHT_MODEL))

    GenerateCaptionModel = ViCaptioningImgModel(version_ResNet=18,vocab_size=vocab.vocab_size, embedding_dim = 512, hidden_size = 512, num_layer_LSTM=2)
    GenerateCaptionModel.load_state_dict(state_dict)

    Trainer = TrainModel()
    return Trainer.generate_caption(model=GenerateCaptionModel, img=img, vocab=vocab, max_length=20)





# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. CSS TÙY CHỈNH (làm giao diện đẹp & chuyên nghiệp hơn)
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }
    .caption-box {
        background-color: #f4f6ff;
        border: 1px solid #dbe2ff;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        font-size: 1.15rem;
        color: #1f2937;
        min-height: 70px;
        line-height: 1.5;
    }
    .caption-placeholder {
        color: #9ca3af;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. TIÊU ĐỀ CHÍNH
# =========================================================
st.markdown('<div class="main-title">🖼️ AI Image Captioning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Tải ảnh lên hoặc chụp ảnh trực tiếp — hệ thống sẽ tự động sinh mô tả (caption) cho ảnh</div>',
    unsafe_allow_html=True,
)


# =========================================================
# 4. HÀM SINH CAPTION (thay bằng model thật của bạn)
# =========================================================
def generate_caption(image: Image.Image) -> str:
    caption = prepare_model_and_generate_caption(img=image)
    time.sleep(1.5)  # giả lập thời gian inference
    return caption


# =========================================================
# 5. KHU VỰC NHẬP ẢNH: Upload hoặc Camera
# =========================================================
st.subheader("1️⃣ Chọn ảnh đầu vào")

tab_upload, tab_camera = st.tabs(["📁 Tải ảnh lên", "📷 Chụp từ camera"])

input_image = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Chọn một file ảnh (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Kéo thả hoặc bấm để chọn file ảnh từ máy tính của bạn",
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

with tab_camera:
    camera_file = st.camera_input("Chụp ảnh bằng camera")
    if camera_file is not None:
        input_image = Image.open(camera_file).convert("RGB")

# Hiển thị ảnh xem trước
if input_image is not None:
    st.image(input_image, caption="Ảnh đã chọn", use_container_width=True)
else:
    st.info("Vui lòng tải ảnh lên hoặc chụp ảnh từ camera để tiếp tục.")


# =========================================================
# 6. NÚT SUBMIT
# =========================================================
st.subheader("2️⃣ Sinh caption")

submit_col, _ = st.columns([1, 3])
with submit_col:
    submit_clicked = st.button(
        "🚀 Generate Caption",
        type="primary",
        use_container_width=True,
        disabled=input_image is None,
    )


# =========================================================
# 7. KHUNG HIỂN THỊ CAPTION
# =========================================================
st.subheader("3️⃣ Kết quả caption")

# Lưu caption vào session_state để không mất khi rerun
if "caption_result" not in st.session_state:
    st.session_state["caption_result"] = ""

if submit_clicked and input_image is not None:
    with st.spinner("Model đang phân tích ảnh và sinh caption..."):
        caption = generate_caption(input_image)
    st.session_state["caption_result"] = caption
    st.success("Đã sinh caption thành công!")

if st.session_state["caption_result"]:
    st.markdown(
        f'<div class="caption-box">{st.session_state["caption_result"]}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="caption-box caption-placeholder">Caption sẽ hiển thị ở đây sau khi bạn bấm "Generate Caption"...</div>',
        unsafe_allow_html=True,
    )

# Nút copy / tải caption (tiện ích thêm)
if st.session_state["caption_result"]:
    st.download_button(
        label="⬇️ Tải caption (.txt)",
        data=st.session_state["caption_result"],
        file_name="caption.txt",
        mime="text/plain",
    )


# =========================================================
# 8. FOOTER
# =========================================================
st.markdown("---")
st.caption("Made with ❤️ using Streamlit · AI Image Captioning System")
