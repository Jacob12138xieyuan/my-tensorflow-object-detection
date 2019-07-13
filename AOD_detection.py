import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2


cap = cv2.VideoCapture("street.mp4")
num = int(cap.get(7))
_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (3, 3), 0)
height, width = first_frame.shape[0:2]
min_score_thresh =0.2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
#from object_detection.utils import ops as utils_ops

from utils import label_map_util
from utils import visualization_utils_AOB as vis_util

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width):
    def reframe_box_masks_to_image_masks_default():
        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
        return tf.image.crop_and_resize(
            image=box_masks_expanded,
            boxes=reverse_boxes,
            box_ind=tf.range(num_boxes),
            crop_size=[image_height, image_width],
            extrapolation_value=0.0)
    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
    return tf.squeeze(image_masks, axis=3)


MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'only_person.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
    with tf.Session() as sess:

        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        for n in range(num):
            ret, image_np = cap.read()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0') 
            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image_np, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            
            boxes = output_dict['detection_boxes']
            classes = output_dict['detection_classes']
            scores = output_dict['detection_scores']
            
            for i in range(boxes.shape[0]):
                if scores is None or (scores[i] > min_score_thresh and classes[i] == 1):
                    box = tuple(boxes[i].tolist())
                    ymin, xmin, ymax, xmax = box
                    ymin, xmin, ymax, xmax = int(ymin*height), int(xmin*width), int(ymax*height), int(xmax*width)
                    #cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    image_np[ymin-30:ymax+30, xmin-30:xmax+30] = first_frame[ymin-30:ymax+30, xmin-30:xmax+30]

            frame_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), 0)
            difference = cv2.absdiff(first_gray, frame_gray)
            _, difference = cv2.threshold(difference, 13, 255, cv2.THRESH_BINARY)

            kernel = np.ones((3,3), np.uint8)
            difference = cv2.erode(difference, kernel, iterations=1) 
            difference = cv2.dilate(difference, kernel, iterations=1) 
            cv2.imshow('diff',difference)

            cv2.imshow('object detection', image_np)#, cv2.resize(image_np,(800,600)))
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        cv2.destroyAllWindows()
        cap.release()

