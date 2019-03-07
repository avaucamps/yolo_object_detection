import colorsys
import os
import random
from keras import backend as K
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_classes(path):
    with open(path) as f:
        classes = f.readlines()

    classes = [c.strip() for c in classes]
    return classes


def read_anchors(path):
    with open(path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1,2)
    
    return anchors


def get_image_shape(image_file):
    image = Image.open(image_file)
    width, height = image.size
    return (height, width)

def scale_boxes(boxes, image_shape):
    '''
    Returns boxes coordinates for image with shape image_shape.

    Arguments:
        boxes: tensor of box coordinates (ymin, xmin, ymax, xmax) with values between 0 and 1
        image_shape: shape of the image the boxes must be scaled for

    Returns:
        boxes: tensor of boxes scaled for the dimensions image_shape
    '''
    height, width = image_shape
    dimensions = np.array([height, width, height, width]).reshape([1,4])
    boxes = boxes * dimensions
    return boxes


def preprocess_image(path, model_input_size):
    image = Image.open(path)
    resized_image = image.resize(model_input_size)
    image_data = np.array(resized_image, dtype="float32")
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) # Add batch dimension

    return image, image_data


def get_font(base_path, image_size):
    font_path = os.path.join(base_path, 'font/FiraMono-Medium.otf')
    font_size = np.floor(3e-2 * image_size[1] + 0.5).astype('int32')
    font = ImageFont.truetype(font=font_path, size=font_size)
    thickness = (image_size[0] + image_size[1]) // 300

    return font, thickness


def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors, font, thickness):
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)
    random.shuffle(colors)
    random.seed(None)
    return colors