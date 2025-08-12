import streamlit as st
import os
from dotenv import load_dotenv
from services import (
    lifestyle_shot_by_image,
    lifestyle_shot_by_text,
    add_shadow,
    create_packshot,
    enhance_prompt,
    generative_fill,
    generate_hd_image,
    erase_foreground
)
from PIL import Image
import io
import requests
import json
import time
from streamlit_drawable_canvas import st_canvas
import numpy as np

st.set_page_config(
    page_title="Quicksnap Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

BRIA_API_KEY = os.getenv("BRIA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  

if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception:
        openai = None
else:
    openai = None

def initialize_session_state():
    """Initialize all session state keys used in the app with safe defaults."""
    defaults = {
        "api_key": BRIA_API_KEY,
        "generated_images": [],
        "current_image": None,
        "pending_urls": [],
        "edited_image": None,
        "original_prompt": "",
        "enhanced_prompt": None,
        "chat_history": [],
        "last_image": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def download_image(url):
    """Download image from URL and return bytes. Safe for None inputs."""
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None

def apply_image_filter(image, filter_type):
    """Apply various filters to the image. Accepts bytes or file-like."""
    try:
        img = Image.open(io.BytesIO(image)) if isinstance(image, (bytes, bytearray)) else Image.open(image)
        if filter_type == "Grayscale":
            return img.convert('L')
        elif filter_type == "Sepia":
            width, height = img.size
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    r, g, b = img.getpixel((x, y))[:3]
                    tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                    tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                    tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                    pixels[x, y] = (min(tr, 255), min(tg, 255), min(tb, 255))
            return img
        elif filter_type == "High Contrast":
            return img.point(lambda x: x * 1.5)
        elif filter_type == "Blur":
            return img.filter(Image.BLUR)
        else:
            return img
    except Exception:
        return None

def check_generated_images():
    """Check pending URLs for readiness. If ready, set edited_image and generated_images."""
    pending = st.session_state.get("pending_urls") or []
    if not pending:
        return False
    ready_images = []
    still_pending = []
    for url in pending:
        try:
            r = requests.head(url, timeout=6)
            if r.status_code == 200:
                ready_images.append(url)
            else:
                still_pending.append(url)
        except Exception:
            still_pending.append(url)
    st.session_state.pending_urls = still_pending
    if ready_images:
        st.session_state.edited_image = ready_images[0]
        if len(ready_images) > 1:
            st.session_state.generated_images = ready_images
        return True
    return False

def auto_check_images(status_container, max_attempts=3):
    """Poll pending_urls a few times for readiness (non-blocking minimal)."""
    attempt = 0
    while attempt < max_attempts and st.session_state.get("pending_urls"):
        time.sleep(1.5)
        if check_generated_images():
            status_container.success("✨ Image ready!")
            return True
        attempt += 1
    return False

def ask_llm_intent(user_input: str) -> dict:
    """
    Use OpenAI to parse intent. Returns a dict with keys:
    - task: "generate"|"shadow"|"lifestyle"|"chat"
    - prompt: optional prompt string
    - reply: optional chat reply for 'chat'
    Robust: if parsing or openai fails, return chat fallback.
    """
    if not openai:
        return {"task": "chat", "reply": user_input}
    system_prompt = (
        "You are an assistant that CAN chat naturally, and CAN output a JSON object "
        "describing an intent for image editing or generation. Respond only with valid JSON.\n\n"
        "If the user requests creating an image output: {\"task\":\"generate\",\"prompt\":\"...\"}\n"
        "If user requests adding a shadow: {\"task\":\"shadow\"}\n"
        "If user requests a lifestyle shot from an existing image: {\"task\":\"lifestyle\",\"prompt\":\"...\"}\n"
        "For normal chat responses: {\"task\":\"chat\",\"reply\":\"...\"}\n"
        "Do NOT include any other text outside the JSON"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            max_tokens=300
        )
        content = resp.choices[0].message.get("content") if hasattr(resp.choices[0].message, "get") else resp.choices[0].message["content"]
        
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            try:
                start = content.index("{")
                end = content.rindex("}") + 1
                parsed = json.loads(content[start:end])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        
        return {"task": "chat", "reply": content.strip() or user_input}
    except Exception:
        return {"task": "chat", "reply": user_input}

def main():
    initialize_session_state()

    st.title("Quicksnap Studio")

    with st.sidebar:
        st.header("Settings")
        entered = st.text_input("Enter your Bria API key:", value=st.session_state.api_key or "", type="password")
        if entered:
            st.session_state.api_key = entered

    tabs = st.tabs([
        "🎨 Generate Image",
        "🖼️ Lifestyle Shot",
        "🎨 Generative Fill",
        "🎨 Erase Elements",
        "💬 Chat Assistant"
    ])

    with tabs[0]:
        st.header("Generate Images")
        col1, col2 = st.columns([2, 1])
        with col1:
            prompt = st.text_area("Enter your prompt", value=st.session_state.get("original_prompt", ""), height=120, key="prompt_input")
            if prompt != st.session_state.get("original_prompt", ""):
                st.session_state.original_prompt = prompt
                st.session_state.enhanced_prompt = None
            if st.session_state.get("enhanced_prompt"):
                st.markdown("**Enhanced Prompt:**")
                st.markdown(f"*{st.session_state.enhanced_prompt}*")
            if st.button("✨ Enhance Prompt"):
                if not prompt:
                    st.warning("Please enter a prompt first.")
                else:
                    res = enhance_prompt(st.session_state.api_key, prompt)
                    if res:
                        st.session_state.enhanced_prompt = res
                        st.success("Prompt enhanced!")
                        st.experimental_rerun()
        with col2:
            num_images = st.slider("Number of images", 1, 4, 1)
            aspect_ratio = st.selectbox("Aspect ratio", ["1:1", "16:9", "9:16", "4:3", "3:4"])
            enhance_img = st.checkbox("Enhance image quality", value=True)
            st.subheader("Style Options")
            style = st.selectbox("Image Style", ["Realistic", "Artistic", "Cartoon", "Sketch", "Watercolor", "Oil Painting", "Digital Art"])
            if style and style != "Realistic":
                prompt_for_api = f"{st.session_state.enhanced_prompt or prompt}, in {style.lower()} style"
            else:
                prompt_for_api = st.session_state.enhanced_prompt or prompt

        if st.button("🎨 Generate Images"):
            if not st.session_state.api_key:
                st.error("Please provide your Bria API key in the sidebar.")
            elif not prompt_for_api:
                st.warning("Please enter a prompt.")
            else:
                with st.spinner("Generating..."):
                    try:
                        result = generate_hd_image(
                            prompt=prompt_for_api,
                            api_key=st.session_state.api_key,
                            num_results=num_images,
                            aspect_ratio=aspect_ratio,
                            sync=True,
                            enhance_image=enhance_img,
                            medium="art" if style != "Realistic" else "photography",
                            prompt_enhancement=False,
                            content_moderation=True
                        )
                        img_url = None
                        if isinstance(result, dict):
                            if "result_url" in result:
                                img_url = result["result_url"]
                            elif "result_urls" in result and result["result_urls"]:
                                img_url = result["result_urls"][0]
                            elif "result" in result and isinstance(result["result"], list):
                                for it in result["result"]:
                                    if isinstance(it, dict) and "urls" in it and it["urls"]:
                                        img_url = it["urls"][0]
                                        break
                                    elif isinstance(it, list) and it:
                                        img_url = it[0]
                                        break
                        if img_url:
                            st.session_state.edited_image = img_url
                            st.success("✨ Image generated successfully!")
                        else:
                            st.error("No result URL found in API response.")
                    except Exception as e:
                        st.error(f"Error generating images: {e}")

        if st.session_state.edited_image:
            st.image(st.session_state.edited_image, caption="Edited Image", use_container_width=True)
            image_data = download_image(st.session_state.edited_image)
            if image_data:
                st.download_button("⬇️ Download Result", image_data, "edited_image.png", "image/png")
        elif st.session_state.pending_urls:
            st.info("Images are being generated. Click the refresh button in Lifestyle or Generative Fill tab to check.")

    with tabs[1]:
        st.header("Product Photography")
        uploaded_file = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"], key="product_upload")
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", use_container_width=True)
                edit_option = st.selectbox("Select Edit Option", ["Create Packshot", "Add Shadow", "Lifestyle Shot"])
                if edit_option == "Create Packshot":
                    bg_color = st.color_picker("Background Color", "#FFFFFF")
                    sku = st.text_input("SKU (optional)", "")
                    force_rmbg = st.checkbox("Force Background Removal", False)
                    content_moderation = st.checkbox("Enable Content Moderation", False)
                    if st.button("Create Packshot"):
                        with st.spinner("Creating packshot..."):
                            try:
                                image_data = uploaded_file.getvalue()
                                if force_rmbg:
                                    try:
                                        from services.background_service import remove_background
                                        bg_result = remove_background(st.session_state.api_key, image_data, content_moderation=content_moderation)
                                        if bg_result and "result_url" in bg_result:
                                            image_data = requests.get(bg_result["result_url"]).content
                                        else:
                                            st.error("Background removal failed.")
                                            image_data = None
                                    except Exception:
                                        pass
                                if image_data:
                                    result = create_packshot(st.session_state.api_key, image_data, background_color=bg_color, sku=sku or None, force_rmbg=force_rmbg, content_moderation=content_moderation)
                                    if result and "result_url" in result:
                                        st.session_state.edited_image = result["result_url"]
                                        st.success("✨ Packshot created!")
                                    else:
                                        st.error("No result from packshot API.")
                            except Exception as e:
                                st.error(f"Error creating packshot: {e}")

                elif edit_option == "Add Shadow":
                    shadow_type = st.selectbox("Shadow Type", ["Natural", "Drop"])
                    bg_color = st.color_picker("Background Color (optional)", "#FFFFFF")
                    use_transparent_bg = st.checkbox("Use Transparent Background", True)
                    shadow_color = st.color_picker("Shadow Color", "#000000")
                    sku = st.text_input("SKU (optional)", "")
                    offset_x = st.slider("X Offset", -50, 50, 0)
                    offset_y = st.slider("Y Offset", -50, 50, 15)
                    shadow_intensity = st.slider("Shadow Intensity", 0, 100, 60)
                    shadow_blur = st.slider("Shadow Blur", 0, 50, 15)
                    force_rmbg = st.checkbox("Force Background Removal", False)
                    content_moderation = st.checkbox("Enable Content Moderation", False)
                    if st.button("Add Shadow"):
                        with st.spinner("Adding shadow..."):
                            try:
                                result = add_shadow(
                                    api_key=st.session_state.api_key,
                                    image_data=uploaded_file.getvalue(),
                                    shadow_type=shadow_type.lower(),
                                    background_color=None if use_transparent_bg else bg_color,
                                    shadow_color=shadow_color,
                                    shadow_offset=[offset_x, offset_y],
                                    shadow_intensity=shadow_intensity,
                                    shadow_blur=shadow_blur,
                                    sku=sku or None,
                                    force_rmbg=force_rmbg,
                                    content_moderation=content_moderation
                                )
                                img_url = None
                                if isinstance(result, dict):
                                    if "result_url" in result:
                                        img_url = result["result_url"]
                                    elif "result_urls" in result and result["result_urls"]:
                                        img_url = result["result_urls"][0]
                                if img_url:
                                    st.session_state.edited_image = img_url
                                    st.success("✨ Shadow added.")
                                else:
                                    st.error("No result URL from shadow API.")
                            except Exception as e:
                                st.error(f"Error adding shadow: {e}")

                elif edit_option == "Lifestyle Shot":
                    shot_type = st.radio("Shot Type", ["Text Prompt", "Reference Image"])
                    placement_type = st.selectbox("Placement Type", ["Original", "Automatic", "Manual Placement", "Manual Padding", "Custom Coordinates"])
                    num_results = st.slider("Number of Results", 1, 8, 4)
                    sync_mode = st.checkbox("Synchronous Mode", False)
                    original_quality = st.checkbox("Original Quality", False)
                    if placement_type == "Manual Placement":
                        positions = st.multiselect("Select Positions", ["Upper Left", "Upper Right", "Bottom Left", "Bottom Right", "Center"], ["Upper Left"])
                    if placement_type == "Manual Padding":
                        pad_left = st.number_input("Left Padding", 0, 1000, 0)
                        pad_right = st.number_input("Right Padding", 0, 1000, 0)
                        pad_top = st.number_input("Top Padding", 0, 1000, 0)
                        pad_bottom = st.number_input("Bottom Padding", 0, 1000, 0)
                    if placement_type in ["Automatic", "Manual Placement", "Custom Coordinates"]:
                        shot_width = st.number_input("Width", 100, 2000, 1000)
                        shot_height = st.number_input("Height", 100, 2000, 1000)

                    if shot_type == "Text Prompt":
                        scene_prompt = st.text_area("Describe the environment")
                        fast_mode = st.checkbox("Fast Mode", True)
                        optimize_desc = st.checkbox("Optimize Description", True)
                        if st.button("Generate Lifestyle Shot") and scene_prompt:
                            with st.spinner("Generating lifestyle shot..."):
                                try:
                                    manual_placements = [p.lower().replace(" ", "_") for p in positions] if placement_type == "Manual Placement" else ["upper_left"]
                                    result = lifestyle_shot_by_text(
                                        api_key=st.session_state.api_key,
                                        image_data=uploaded_file.getvalue(),
                                        scene_description=scene_prompt,
                                        placement_type=placement_type.lower().replace(" ", "_"),
                                        num_results=num_results,
                                        sync=sync_mode,
                                        fast=fast_mode,
                                        optimize_description=optimize_desc,
                                        shot_size=[shot_width, shot_height] if placement_type != "Original" else [1000, 1000],
                                        original_quality=original_quality,
                                        exclude_elements=None if fast_mode else None,
                                        manual_placement_selection=manual_placements,
                                        padding_values=[pad_left, pad_right, pad_top, pad_bottom] if placement_type == "Manual Padding" else [0,0,0,0],
                                        foreground_image_size=None,
                                        foreground_image_location=None,
                                        force_rmbg=False,
                                        content_moderation=False,
                                        sku=None
                                    )
                                    img_url = None
                                    if isinstance(result, dict):
                                        if "result_url" in result:
                                            img_url = result["result_url"]
                                        elif "result_urls" in result and result["result_urls"]:
                                            img_url = result["result_urls"][0]
                                    if img_url:
                                        st.session_state.edited_image = img_url
                                        st.success("✨ Lifestyle shot generated!")
                                    else:
                                        st.error("No result URL from lifestyle API.")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    else:
                        ref_image = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"], key="ref_upload")
                        if st.button("Generate Lifestyle Shot") and ref_image:
                            with st.spinner("Generating lifestyle shot..."):
                                try:
                                    manual_placements = [p.lower().replace(" ", "_") for p in positions] if placement_type == "Manual Placement" else ["upper_left"]
                                    result = lifestyle_shot_by_image(
                                        api_key=st.session_state.api_key,
                                        image_data=uploaded_file.getvalue(),
                                        reference_image=ref_image.getvalue(),
                                        placement_type=placement_type.lower().replace(" ", "_"),
                                        num_results=num_results,
                                        sync=sync_mode,
                                        shot_size=[shot_width, shot_height] if placement_type != "Original" else [1000, 1000],
                                        original_quality=original_quality,
                                        manual_placement_selection=manual_placements,
                                        padding_values=[pad_left, pad_right, pad_top, pad_bottom] if placement_type == "Manual Padding" else [0,0,0,0],
                                        foreground_image_size=None,
                                        foreground_image_location=None,
                                        force_rmbg=False,
                                        content_moderation=False,
                                        sku=None,
                                        enhance_ref_image=True,
                                        ref_image_influence=1.0
                                    )
                                    img_url = None
                                    if isinstance(result, dict):
                                        if "result_url" in result:
                                            img_url = result["result_url"]
                                        elif "result_urls" in result and result["result_urls"]:
                                            img_url = result["result_urls"][0]
                                    if img_url:
                                        st.session_state.edited_image = img_url
                                        st.success("✨ Lifestyle shot generated!")
                                    else:
                                        st.error("No result URL from lifestyle API.")
                                except Exception as e:
                                    st.error(f"Error: {e}")

            with col2:
                if st.session_state.edited_image:
                    st.image(st.session_state.edited_image, caption="Edited Image", use_container_width=True)
                    image_data = download_image(st.session_state.edited_image)
                    if image_data:
                        st.download_button("⬇️ Download Result", image_data, "edited_product.png", "image/png")
                elif st.session_state.pending_urls:
                    st.info("Images are being generated. Use refresh controls in other tabs to check status.")

    with tabs[2]:
        st.header("🎨 Generative Fill")
        st.markdown("Draw a mask on the image and describe what you want to generate in that area.")
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="fill_upload")
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", use_container_width=True)
                try:
                    img = Image.open(uploaded_file)
                    img_width, img_height = img.size
                    aspect_ratio = img_height / img_width
                    canvas_width = min(img_width, 800)
                    canvas_height = int(canvas_width * aspect_ratio)
                    img = img.resize((canvas_width, canvas_height))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Convert PIL Image to bytes for st_canvas
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    st.write(f"Debug: Image mode: {img.mode}, Size: {img.size}")  # Debugging output
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
                    return

                stroke_width = st.slider("Brush width", 1, 50, 20)
                stroke_color = st.color_picker("Brush color", "#fff")

                try:
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        background_image=img_bytes,
                        height=canvas_height,
                        width=canvas_width,
                        drawing_mode="freedraw",
                        key="canvas",
                    )
                except Exception as e:
                    st.error(f"Error initializing canvas: {e}")
                    return

                prompt = st.text_area("Describe what to generate in the masked area")
                negative_prompt = st.text_area("Describe what to avoid (optional)")
                col_a, col_b = st.columns(2)
                with col_a:
                    num_results = st.slider("Number of variations", 1, 4, 1)
                    sync_mode = st.checkbox("Synchronous Mode", False, key="gen_fill_sync_mode")
                with col_b:
                    seed = st.number_input("Seed (optional)", min_value=0, value=0)
                    content_moderation = st.checkbox("Enable Content Moderation", False, key="gen_fill_content_mod")
                if st.button("🎨 Generate", type="primary"):
                    if not prompt:
                        st.error("Please enter a prompt describing what to generate.")
                    elif canvas_result.image_data is None:
                        st.error("Please draw a mask on the image first.")
                    else:
                        mask_img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA').convert('L')
                        mask_bytes = io.BytesIO()
                        mask_img.save(mask_bytes, format='PNG')
                        mask_bytes = mask_bytes.getvalue()
                        image_bytes = uploaded_file.getvalue()
                        with st.spinner("🎨 Generating..."):
                            try:
                                result = generative_fill(
                                    st.session_state.api_key,
                                    image_bytes,
                                    mask_bytes,
                                    prompt,
                                    negative_prompt=negative_prompt or None,
                                    num_results=num_results,
                                    sync=sync_mode,
                                    seed=seed if seed != 0 else None,
                                    content_moderation=content_moderation
                                )
                                img_url = None
                                if sync_mode:
                                    if isinstance(result, dict):
                                        if "urls" in result and result["urls"]:
                                            img_url = result["urls"][0]
                                        elif "result_url" in result:
                                            img_url = result["result_url"]
                                else:
                                    if isinstance(result, dict):
                                        if "urls" in result:
                                            st.session_state.pending_urls = result["urls"][:num_results]
                                        elif "result" in result and isinstance(result["result"], list):
                                            urls = []
                                            for item in result["result"]:
                                                if isinstance(item, dict) and "urls" in item:
                                                    urls.extend(item["urls"])
                                                elif isinstance(item, list):
                                                    urls.extend(item)
                                                if len(urls) >= num_results:
                                                    break
                                            st.session_state.pending_urls = urls[:num_results]
                                    if st.session_state.pending_urls:
                                        status_container = st.empty()
                                        status_container.info(f"🎨 Generation started! Waiting for {len(st.session_state.pending_urls)} image(s)...")
                                        if auto_check_images(status_container):
                                            st.experimental_rerun()
                                if img_url:
                                    st.session_state.edited_image = img_url
                                    st.success("✨ Generation complete!")
                            except Exception as e:
                                st.error(f"Error: {e}")

            with col2:
                if st.session_state.edited_image:
                    st.image(st.session_state.edited_image, caption="Generated Result", use_container_width=True)
                    img_bytes = download_image(st.session_state.edited_image)
                    if img_bytes:
                        st.download_button("⬇️ Download Result", img_bytes, "generated_fill.png", "image/png")
                elif st.session_state.pending_urls:
                    st.info("Generation in progress. Click the refresh button above to check status.")

    with tabs[3]:
        st.header("🎨 Erase Elements")
        st.markdown("Upload an image and select the area you want to erase.")
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="erase_upload")
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", use_container_width=True)
                try:
                    img = Image.open(uploaded_file)
                    img_width,

 img_height = img.size
                    aspect_ratio = img_height / img_width
                    canvas_width = min(img_width, 800)
                    canvas_height = int(canvas_width * aspect_ratio)
                    img = img.resize((canvas_width, canvas_height))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Convert PIL Image to bytes for st_canvas
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    st.write(f"Debug: Image mode: {img.mode}, Size: {img.size}")  # Debugging output
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
                    return

                stroke_width = st.slider("Brush width", 1, 50, 20, key="erase_brush_width")
                stroke_color = st.color_picker("Brush color", "#fff", key="erase_brush_color")
                try:
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 0.0)",
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        background_image=img_bytes,
                        drawing_mode="freedraw",
                        height=canvas_height,
                        width=canvas_width,
                        key="erase_canvas"
                    )
                except Exception as e:
                    st.error(f"Error initializing canvas: {e}")
                    return

                content_moderation = st.checkbox("Enable Content Moderation", False, key="erase_content_mod")
                if st.button("🎨 Erase Selected Area", key="erase_btn"):
                    if canvas_result.image_data is None:
                        st.warning("Please draw on the image to select the area to erase.")
                    else:
                        with st.spinner("Erasing selected area..."):
                            try:
                                image_bytes = uploaded_file.getvalue()
                                result = erase_foreground(
                                    st.session_state.api_key,
                                    image_data=image_bytes,
                                    content_moderation=content_moderation
                                )
                                img_url = None
                                if isinstance(result, dict) and "result_url" in result:
                                    img_url = result["result_url"]
                                if img_url:
                                    st.session_state.edited_image = img_url
                                    st.success("✨ Area erased successfully!")
                                else:
                                    st.error("No result URL from erase API.")
                            except Exception as e:
                                st.error(f"Error: {e}")
            with col2:
                if st.session_state.edited_image:
                    st.image(st.session_state.edited_image, caption="Result", use_container_width=True)
                    img_bytes = download_image(st.session_state.edited_image)
                    if img_bytes:
                        st.download_button("⬇️ Download Result", img_bytes, "erased_image.png", "image/png", key="erase_download")

    with tabs[4]:
        st.header("💬 AI Chat Assistant")
        st.markdown("Talk normally. I can chat and also create/edit images.")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg.get("role", "assistant")):
                if msg.get("type") == "image":
                    st.image(msg.get("content"), caption=msg.get("caption", "Image"), use_container_width=True)
                else:
                    st.markdown(msg.get("content", ""))

        user_input = st.chat_input("Type your request here...")
        if not user_input:
            return  

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        intent = {"task": "chat", "reply": user_input}
        if openai:
            intent = ask_llm_intent(user_input)
        else:
            t = user_input.lower()
            if "generate" in t or "create" in t:
                intent = {"task": "generate", "prompt": user_input}
            elif "shadow" in t:
                intent = {"task": "shadow"}
            elif "lifestyle" in t:
                intent = {"task": "lifestyle", "prompt": user_input}
            else:
                intent = {"task": "chat", "reply": user_input}

        task = intent.get("task", "chat")

        if task == "generate":
            prompt_text = intent.get("prompt", user_input)
            with st.chat_message("assistant"):
                st.markdown("🎨 Generating your image...")
            try:
                result = generate_hd_image(
                    prompt=prompt_text,
                    api_key=st.session_state.api_key,
                    num_results=1,
                    aspect_ratio="1:1",
                    sync=True,
                    enhance_image=True,
                    medium="photography",
                    prompt_enhancement=True,
                    content_moderation=True
                )
                img_url = None
                if isinstance(result, dict):
                    if "result_url" in result:
                        img_url = result["result_url"]
                    elif "result_urls" in result and result["result_urls"]:
                        img_url = result["result_urls"][0]
                    elif "result" in result and isinstance(result["result"], list):
                        for it in result["result"]:
                            if isinstance(it, dict) and "urls" in it and it["urls"]:
                                img_url = it["urls"][0]
                                break
                    if img_url:
                        st.session_state.last_image = img_url
                        img_bytes = download_image(img_url)
                        if img_bytes:
                            st.session_state.chat_history.append({"role": "assistant", "type": "image", "content": img_bytes})
                            with st.chat_message("assistant"):
                                st.image(img_bytes, caption="Generated Image", use_container_width=True)
                        else:
                            with st.chat_message("assistant"):
                                st.markdown("✅ Image generated but failed to download preview. You can download it from the result URL.")
                    else:
                        with st.chat_message("assistant"):
                            st.markdown("❌ Could not generate the image (no URL returned).")
            except Exception as e:
                with st.chat_message("assistant"):
                    st.markdown(f"⚠️ Error: {e}")

        elif task == "shadow":
            if not st.session_state.last_image:
                with st.chat_message("assistant"):
                    st.markdown("⚠️ No last image available. Generate or upload an image first.")
            else:
                with st.chat_message("assistant"):
                    st.markdown("🖌️ Adding shadow to last image...")
                img_bytes = download_image(st.session_state.last_image)
                if not img_bytes:
                    with st.chat_message("assistant"):
                        st.markdown("⚠️ Failed to download last image for editing.")
                else:
                    try:
                        result = add_shadow(
                            api_key=st.session_state.api_key,
                            image_data=img_bytes,
                            shadow_type="drop",
                            background_color="#FFFFFF",
                            shadow_color="#000000",
                            shadow_offset=[0, 15],
                            shadow_intensity=60,
                            shadow_blur=15
                        )
                        img_url = None
                        if isinstance(result, dict):
                            if "result_url" in result:
                                img_url = result["result_url"]
                            elif "result_urls" in result and result["result_urls"]:
                                img_url = result["result_urls"][0]
                        if img_url:
                            st.session_state.last_image = img_url
                            new_bytes = download_image(img_url)
                            if new_bytes:
                                st.session_state.chat_history.append({"role": "assistant", "type": "image", "content": new_bytes})
                                with st.chat_message("assistant"):
                                    st.image(new_bytes, caption="Image with Shadow", use_container_width=True)
                        else:
                            with st.chat_message("assistant"):
                                st.markdown("❌ Shadow API returned no URL.")
                    except Exception as e:
                        with st.chat_message("assistant"):
                            st.markdown(f"⚠️ Error: {e}")

        elif task == "lifestyle":
            if not st.session_state.last_image:
                with st.chat_message("assistant"):
                    st.markdown("⚠️ No last image available. Generate or upload an image first.")
            else:
                prompt_text = intent.get("prompt", user_input)
                with st.chat_message("assistant"):
                    st.markdown("🏠 Creating lifestyle shot...")
                img_bytes = download_image(st.session_state.last_image)
                if not img_bytes:
                    with st.chat_message("assistant"):
                        st.markdown("⚠️ Failed to download last image for editing.")
                else:
                    try:
                        result = lifestyle_shot_by_text(
                            api_key=st.session_state.api_key,
                            image_data=img_bytes,
                            scene_description=prompt_text,
                            placement_type="automatic",
                            num_results=1,
                            sync=True,
                            fast=True,
                            optimize_description=True,
                            shot_size=[1000, 1000],
                            original_quality=True
                        )
                        img_url = None
                        if isinstance(result, dict):
                            if "result_url" in result:
                                img_url = result["result_url"]
                            elif "result_urls" in result and result["result_urls"]:
                                img_url = result["result_urls"][0]
                        if img_url:
                            st.session_state.last_image = img_url
                            new_bytes = download_image(img_url)
                            if new_bytes:
                                st.session_state.chat_history.append({"role": "assistant", "type": "image", "content": new_bytes})
                                with st.chat_message("assistant"):
                                    st.image(new_bytes, caption="Lifestyle Shot", use_container_width=True)
                        else:
                            with st.chat_message("assistant"):
                                st.markdown("❌ Lifestyle API returned no URL.")
                    except Exception as e:
                        with st.chat_message("assistant"):
                            st.markdown(f"⚠️ Error: {e}")

        else:
            reply = intent.get("reply", "I'm here to chat or help with images!")
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

if __name__ == "__main__":
    main()
