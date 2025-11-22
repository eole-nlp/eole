import os
import fitz
import img2pdf
import io
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from eole.inference_engine import InferenceEnginePY
from eole.config.run import PredictConfig

# from torch.profiler import profile, record_function, ProfilerActivity


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []

    pdf_document = fitz.open(pdf_path)

    zoom = dpi / 72.0
    # 144/ 72 will make image 1191 x 1684 per page
    # reized after to 1024x1024 by Eole
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background

        images.append(img)

    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images, output_path):

    if not pil_images:
        return

    image_bytes_list = []

    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")


def re_match(text):
    # pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)(.*?)(?=<\|ref\|>|$)"

    matches = re.findall(pattern, text, re.DOTALL)

    match_ref = []
    matches_image = []
    matches_other = []
    content = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
        match_ref.append((a_match[0], a_match[1], a_match[2]))
        content.append(a_match[3])
    return match_ref, matches_image, matches_other, content


def extract_coordinates_and_label(ref_text, image_width, image_height):

    try:
        label_type = ref_text[1]
        # cor_list = eval(ref_text[2].replace(',,', ',').replace('1132', '132'))
        cor_list = eval(ref_text[2])
    except Exception as e:
        print("Bad coord from inference:", e, repr(ref_text))
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx):

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                label_type = "".join(ch for ch in label_type if ch.isprintable())

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20,)

                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == "image":
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{OUTPUT_PATH}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width, text_y + text_height], fill=(255, 255, 255, 30)
                        )
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception as e:
                        print("draw error:", e, repr(label_type))
                        pass
        except Exception as e:
            print("DRAW ERROR:", e, repr(label_type))
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, jdx):
    result_image = draw_bounding_boxes(image, ref_texts, jdx)
    return result_image


if __name__ == "__main__":

    OUTPUT_PATH = "/mnt/InternalCrucial4/LLM_work/deepseek-ocr/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)

    print(f"{Colors.RED}PDF loading .....{Colors.RESET}")

    INPUT_PATH = "/mnt/InternalCrucial4/LLM_work/deepseek-ocr/deepseekocr.pdf"
    images = pdf_to_images_high_quality(INPUT_PATH)

    config = PredictConfig(
        model_path="/mnt/InternalCrucial4/LLM_work/deepseek-ocr",
        src="dummy",
        max_length=8192,
        gpu_ranks=[0],
        compute_dtype="bf16",
        self_attn_backend="flash",
        top_p=0.0,
        top_k=1,
        temperature=0.0,
        beam_size=1,
        seed=12,
        batch_size=64,
        batch_type="sents",
        report_time=True,
        fuse_kvq=False,
        fuse_gate=True,
        # block_ngram_repeat=5,
    )

    config.data_type = "image"

    engine = InferenceEnginePY(config)

    model_input = []
    for i in range(len(images)):
        model_input.append(
            {
                "text": "{image}\n<|grounding|>Convert the document to markdown.",
                "images": {"image": images[i]},
            }
        )
        model_input.append(
            {
                "text": "{image}\n<|grounding|>Convert the document to markdown.",
                "images": {"image": images[i]},
            }
        )

    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("Infer"):
            pred = engine.infer_list(model_input)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
    """

    pred = engine.infer_list(model_input)
    outputs_list = []
    for i in range(len(model_input)):
        # pred tuple (score, estim, preds), preds batch of nbest hence 0
        outputs_list.append(pred[2][i][0].replace("｟newline｠", "\n"))

    output_path = OUTPUT_PATH
    os.makedirs(output_path, exist_ok=True)

    mmd_det_path = output_path + "/" + INPUT_PATH.split("/")[-1].replace(".pdf", "_det.mmd")
    mmd_path = output_path + "/" + INPUT_PATH.split("/")[-1].replace("pdf", "mmd")
    pdf_out_path = output_path + "/" + INPUT_PATH.split("/")[-1].replace(".pdf", "_layouts.pdf")
    contents_det = ""
    contents = ""
    draw_images = []
    jdx = 0
    for content, img in zip(outputs_list, images):

        page_num = "\n<--- Page Split --->"

        contents_det += content + f"\n{page_num}\n"

        image_draw = img.copy()

        matches_ref, matches_images, matches_other, actual_content = re_match(content)
        actual_content = [x for x in actual_content if x != "\n"]
        if len(matches_other) != len(actual_content):
            print(matches_other)
            print("####")
            print(actual_content)
            print("#######################################")

        result_image = process_image_with_refs(image_draw, matches_ref, jdx)
        draw_images.append(result_image)
        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f"![](images/{jdx}_{idx}.jpg)\n")

        for idx, a_match_other in enumerate(matches_other):
            # print(a_match_other)
            content = (
                content.replace(a_match_other, "")
                .replace("\\coloneqq", ":=")
                .replace("\\eqqcolon", "=:")
                .replace("\n\n\n\n", "\n\n")
                .replace("\n\n\n", "\n\n")
            )

        contents += content + f"\n{page_num}\n"

        jdx += 1

    with open(mmd_det_path, "w", encoding="utf-8") as afile:
        afile.write(contents_det)

    with open(mmd_path, "w", encoding="utf-8") as afile:
        afile.write(contents)

    pil_to_pdf_img2pdf(draw_images, pdf_out_path)
