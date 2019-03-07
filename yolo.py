from yad2k.models.keras_yolo import yolo_boxes_to_corners
import keras.backend as K
import tensorflow as tf
from utils import scale_boxes


def get_best_class(box_scores):
    '''
    Get the best class for each box.

    Arguments:
        box_scores: tensor with the score of each class for each box

    Returns:
        box_class_indexes: the index of the best class for each box
        box_class_scores = the score of the best class for each box
    '''
    box_class_indexes = K.argmax(box_scores, -1)
    box_class_scores = K.max(box_scores, -1)

    return box_class_indexes, box_class_scores


def filter_boxes(box_confidence, boxes, box_class_probs, threshold = .4):
    """
    Filters detected objects with class confidence less than threshold. 
    
    Arguments:
    box_confidence: tensor of shape (19, 19, 5, 1).
    boxes: tensor of shape (19, 19, 5, 4).
    box_class_probs: tensor of shape (19, 19, 5, 80).
    threshold: all objects with class confidence below this threshold will be forgotten. 
    
    Returns:
        scores: tensor of shape (None,), containing the class probability score for selected boxes.
        boxes: tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes.
        classes: tensor of shape (None,), containing the index of the class detected by the selected boxes.
    """
    box_scores = box_confidence * box_class_probs
    box_class_indexes, box_class_scores = get_best_class(box_scores)
    filtering_mask = box_class_scores > threshold
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_class_indexes, filtering_mask)

    return scores, boxes, classes


def get_elements_at_indexes(all_elements, indexes):
    return K.gather(all_elements, indexes)


def non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold):
    """
    Applies Non-max suppression to set of boxes.
    
    Arguments:
        scores: tensor of shape (None,).
        boxes: tensor of shape (None, 4).
        classes: tensor of shape (None,).
        max_boxes: integer, maximum number of predicted boxes.
        iou_threshold: real value, "intersection over union" threshold used for NMS filtering.
    
    Returns:
        scores: tensor of shape (, None), predicted score for each box.
        boxes: tensor of shape (4, None), predicted box coordinates.
        classes: tensor of shape (, None), predicted class for each box.
    """
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    indexes = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)
    scores = get_elements_at_indexes(scores, indexes)
    boxes = get_elements_at_indexes(boxes, indexes)
    classes = get_elements_at_indexes(classes, indexes)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape, max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding to the corresponding boxes with their predicted score, box coordinates and class.
    
    Arguments:
        yolo_outputs: output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                        box_confidence: tensor of shape (None, 19, 19, 5, 1)
                        box_xy: tensor of shape (None, 19, 19, 5, 2)
                        box_wh: tensor of shape (None, 19, 19, 5, 2)
                        box_class_probs: tensor of shape (None, 19, 19, 5, 80)
        image_shape: default shape of the image.
        max_boxes: integer, maximum number of predicted boxes.
        score_threshold: real value, will forget detected objects with score inferior to this value.
        iou_threshold: real value, "intersection over union" threshold used for NMS filtering.
    
    Returns:
        scores: tensor of shape (None, ), predicted score for each box.
        boxes: tensor of shape (None, 4), predicted coordinates for each box.
        classes: tensor of shape (None,), predicted class for each box.
    """
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh) # now box coordinates = (ymin, xmin, ymax, xmax)
    scores, boxes, classes = filter_boxes(box_confidence, boxes, box_class_probs)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes