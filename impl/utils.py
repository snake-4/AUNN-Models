import torch
from PIL import Image

def binary_encode_tensor(x: torch.Tensor, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def binary_arange(startIdx, endIdx, bits, device):
    input1 = torch.arange(startIdx, endIdx, dtype=torch.int64, device=device)
    return binary_encode_tensor(input1, bits).type(torch.get_default_dtype())


def pillow_generate_canvas(images: list[Image.Image], row_length):
    canvas_width = row_length * images[0].width
    canvas_height = (len(images) // row_length) * images[0].height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    # Paste the images onto the canvas
    x_offset = 0
    y_offset = 0
    for img in images:
        if img.size != images[0].size:
            img = img.resize((images[0].width, images[0].height))
        canvas.paste(img, (x_offset, y_offset))
        x_offset += img.width
        if x_offset >= canvas_width:
            x_offset = 0
            y_offset += img.height
    return canvas
