import os
from utils import read_classes, read_anchors, preprocess_image, generate_colors, draw_boxes, get_font, get_image_shape
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yolo import yolo_eval
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import keras.backend as K


def predict(sess, scores, boxes, classes, model, image_file, class_names):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
        sess: tensorflow/Keras session containing the YOLO graph.
        scores: 
        boxes:
        classes:
        model: loaded yolo model.
        image_file: path to the image to predict.
        class_names: names of the classes to predict.
    
    Returns:
        out_scores: tensor of shape (None, ), scores of the predicted boxes.
        out_boxes: tensor of shape (None, 4), coordinates of the predicted boxes.
        out_classes: tensor of shape (None, ), class index of the predicted boxes.
    """
    image, image_data = preprocess_image(image_file, model_input_size=(608, 608))
    out_scores, out_boxes, out_classes = sess.run(
        [scores, boxes, classes], 
        feed_dict={model.input: image_data, K.learning_phase(): 0}
    )

    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    colors = generate_colors(class_names)
    font, thickness = get_font(base_path, image.size)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors, font, thickness)
    plt.imshow(image)
    plt.show()

    return out_scores, out_boxes, out_classes


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))
        
    classes_path = os.path.join(base_path, "coco_classes.txt")
    class_names = read_classes(classes_path)

    anchors_path = os.path.join(base_path, "yolo_anchors.txt")
    anchors = read_anchors(anchors_path)

    image_file = os.path.join(base_path, "images\\test1.jpg")
    image_shape = get_image_shape(image_file)

    model_path = os.path.join(base_path, 'yolo.h5')
    model = load_model(model_path)
    outputs = yolo_head(model.output, anchors, len(class_names))

    scores, boxes, classes = yolo_eval(outputs, image_shape)

    sess = K.get_session()
    out_scores, out_boxes, out_classes = predict(sess, scores, boxes, classes, model, image_file, class_names)