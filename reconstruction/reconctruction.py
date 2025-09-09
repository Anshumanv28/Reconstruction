import os
import json
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

# ------------------------------------------------------------------------------
# Helper functions for text wrapping and font sizing

def get_wrapped_text(text, font, max_width, draw):
    """
    Wraps text to fit within a given pixel width.
    Splits on newlines and wraps each line.
    """
    lines = []
    for line in text.splitlines():
        if line.strip() == "":
            continue
        words = line.split()
        if not words:
            continue
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return "\n".join(lines)


def get_max_font_for_box(draw, bbox, text, max_font_size=40, min_font_size=8,
                        font_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "SakalBharati.ttf")):
    """
    Determines the maximum font size that allows the text to fit inside the bounding box using binary search.
    Returns the font size and the wrapped text.
    """
    x_min, y_min, x_max, y_max = bbox
    box_width = x_max - x_min
    box_height = y_max - y_min
    max_font_size = int(box_height)

    def fits(font_size):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
        wrapped = get_wrapped_text(text, font, box_width, draw)
        bbox_text = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        return (text_width <= box_width and text_height <= box_height), wrapped

    low, high = min_font_size, max_font_size
    best_font_size = min_font_size
    try:
        font = ImageFont.truetype(font_path, min_font_size)
    except IOError:
        font = ImageFont.load_default()
    best_wrapped = get_wrapped_text(text, font, box_width, draw)

    while low <= high:
        mid = (low + high) // 2
        does_fit, wrapped_text = fits(mid)
        if does_fit:
            best_font_size = mid
            best_wrapped = wrapped_text
            low = mid + 1
        else:
            high = mid - 1

    return best_font_size, best_wrapped


def draw_text_in_box(draw, bbox, text, font, spacing=4):
    """
    Draws text into the given bounding box using the provided font.
    """
    x_min, y_min, _, _ = bbox
    wrapped = get_wrapped_text(text, font, bbox[2] - bbox[0], draw)
    draw.multiline_text((x_min, y_min), wrapped, fill="black", font=font, spacing=spacing)

# ------------------------------------------------------------------------------
# Table overlay helpers

def get_min_uniform_font_for_table(draw, table_bbox, table_layout, font_path, translate=False):
    """
    For all cells in a table, find the minimum font size that fits the text in each cell's box.
    Returns the minimum font size and a list of wrapped texts for each cell.
    """
    min_font_size = None
    wrapped_texts = []
    for cell in table_layout:
        cell_bbox = cell.get("bbox")
        abs_bbox = [
            table_bbox[0] + cell_bbox[0],
            table_bbox[1] + cell_bbox[1],
            table_bbox[0] + cell_bbox[2],
            table_bbox[1] + cell_bbox[3]
        ]
        text = cell.get("translated_text" if translate else "ocr_text", "")
        font_size, wrapped = get_max_font_for_box(draw, abs_bbox, text, font_path=font_path)
        wrapped_texts.append((abs_bbox, text, wrapped))
        if min_font_size is None or font_size < min_font_size:
            min_font_size = font_size
    return min_font_size, wrapped_texts


def draw_table_overlay(image, draw, table_bbox, table_layout, font_path, translate=False):
    """
    Blurs the table area, then overlays each cell with uniform font size and bounding box.
    """
    if len(table_layout) == 0:
        return
    region = image.crop((int(table_bbox[0]), int(table_bbox[1]), int(table_bbox[2]), int(table_bbox[3])))
    blurred = region.filter(ImageFilter.GaussianBlur(radius=35))
    enhancer = ImageEnhance.Brightness(blurred)
    brightened = enhancer.enhance(1.25)
    overlay = Image.new("RGBA", brightened.size, (255, 255, 255, 32))
    brightened_rgba = brightened.convert("RGBA")
    combined = Image.alpha_composite(brightened_rgba, overlay).convert("RGB")
    image.paste(combined, box=(int(table_bbox[0]), int(table_bbox[1])))

    temp_draw = ImageDraw.Draw(image)
    min_font_size, wrapped_texts = get_min_uniform_font_for_table(temp_draw, table_bbox, table_layout, font_path, translate)
    try:
        font = ImageFont.truetype(font_path, min_font_size)
    except IOError:
        font = ImageFont.load_default()

    for idx, cell in enumerate(table_layout):
        cell_bbox = cell.get("bbox")
        abs_bbox = [
            table_bbox[0] + cell_bbox[0],
            table_bbox[1] + cell_bbox[1],
            table_bbox[0] + cell_bbox[2],
            table_bbox[1] + cell_bbox[3]
        ]
        text = cell.get("translated_text" if translate else "ocr_text", "")
        temp_draw.rectangle(abs_bbox, outline="black", width=2)
        if text.strip():
            draw_text_in_box(temp_draw, abs_bbox, text, font)

# ------------------------------------------------------------------------------
# Main document reconstruction function

def reconstruct_document(layout_json, pages_folder, output_pdf_path,
                         font_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "SakalBharati.ttf"),
                         translate=False):
    """
    Reconstructs a PDF document by overlaying OCR (or translated) text in bounding boxes
    on the original page images.
    """
    pages = []
    for page in layout_json.get("pages", []):
        page_number = page.get("page_number", 1)
        page_image_path = os.path.join(pages_folder, f"page_{page_number}.png")
        if not os.path.exists(page_image_path):
            print(f"Page image not found: {page_image_path}")
            continue
        img = Image.open(page_image_path).convert("RGB")
        width, height = img.size

        # Adding a footer
        if translate:
            footer_height = 30
            height += footer_height*2
            footer_bbox = (20, height - footer_height, width, height-footer_height)

        # Create a blank canvas for drawing
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        if translate:
            fnt = ImageFont.load_default()
            draw_text_in_box(draw, footer_bbox, "Disclaimer: This is a machine-translated document. Please cross-check it with the original", fnt)

        for element in page.get("metadata", []):
            if element.get("label") == "table" and "table_layout" in element:
                table_bbox = element.get("bounding_box")
                table_layout = element.get("table_layout", [])
                draw_table_overlay(image, draw, table_bbox, table_layout, font_path, translate)
            elif element.get("label") != "figure":
                bbox = element.get("bounding_box")
                text = element.get("translated_text" if translate else "ocr_text", "") or ""
                if bbox and text.strip():
                    # Simply draw text without any background
                    font_size, wrapped = get_max_font_for_box(draw, bbox, text, font_path=font_path)
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                    except IOError:
                        font = ImageFont.load_default()
                    draw_text_in_box(draw, bbox, text, font)
            else:
                # Paste figures directly
                bbox = element.get("bounding_box")
                bbox_int = [int(coord) for coord in bbox]
                figure_crop = img.crop(tuple(bbox_int))
                image.paste(figure_crop, tuple(bbox_int))

        pages.append(image)

    if pages:
        pages[0].save(output_pdf_path, "PDF", resolution=100.0, save_all=True, append_images=pages[1:])
        print(f"Document saved as PDF: {output_pdf_path}")
    else:
        print("No pages processed. PDF not created.")
