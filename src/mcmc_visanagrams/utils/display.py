import numpy as np


def clip(val, max_val):
    return min(max(val, 0), max_val)


def pad(val, padding, max):
    val_min = clip(val - padding, max)
    val_max = clip(val + padding, max)

    return val_min, val_max


def draw_box(img, start, size, color, max_size):
    padding = 5
    H, W, _ = img.shape

    xmin, ymin = start

    xmax = xmin + size
    ymax = ymin + size

    xmin_lower, xmin_upper = pad(xmin, padding, max_size)
    xmax_lower, xmax_upper = pad(xmax, padding, max_size)
    ymin_lower, ymin_upper = pad(ymin, padding, max_size)
    ymax_lower, ymax_upper = pad(ymax, padding, max_size)

    # Draw each portion of an image
    img[ymin_lower:ymin_upper, xmin:xmax] = np.array(color)[None, None, :]
    img[ymax_lower:ymax_upper, xmin:xmax] = np.array(color)[None, None, :]
    img[ymin:ymax, xmin_lower:xmin_upper] = np.array(color)[None, None, :]
    img[ymin:ymax, xmax_lower:xmax_upper] = np.array(color)[None, None, :]


def visualize_context(canvas_size, base_size, context, color_lookup):
    img = np.zeros([canvas_size, canvas_size, 3])

    for k, v in context.items():
        scale, xstart, ystart = k
        caption = v['string']
        color = color_lookup[caption][0]
        draw_box(img, (xstart, ystart), scale * base_size, color, canvas_size - 1)

    return img
