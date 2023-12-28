import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import zipfile
import math
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt 
plt.switch_backend('Agg')
from PIL import Image

import cv2



video = cv2.VideoCapture("src/cam3.mp4")
width=video.get(cv2.CAP_PROP_FRAME_WIDTH)
height=video.get(cv2.CAP_PROP_FRAME_HEIGHT)
def ROI(image):
    vertix = [
        (0,height),
        (width/2,height*0.55),
        (width,height)
    ]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([vertix],np.int32), (255,255,255))
    Image = cv2.bitwise_and(image, mask)
    return Image




sys.path.append("..")




from utils import label_map_util

from utils import visualization_utils as vis_util



PATH_TO_CKPT ='frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90




detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)




def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



frame_width = int(video.get(3))
frame_height = int(video.get(4))
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret,cap = video.read()
      gray = cv2.cvtColor(cap, cv2.COLOR_RGB2GRAY)
      Edge = cv2.Canny(gray, 200, 300)

    
      line = cv2.HoughLinesP(ROI(Edge),6,np.pi/160, 150, lines=np.array([]),minLineLength=40,maxLineGap=20)
      left_x=[]
      left_y=[]
      right_x=[]
      right_y=[]
      if line is not None:
        for i in line:
            x_1, y_1, x_2, y_2 = i[0]
            slope=(y_2-y_1)/(x_2-x_1)
            if math.fabs(slope) <0.52:
              continue 
            if slope <=0:
              left_x.extend([x_1,x_2])
              left_y.extend([y_1,y_2])
            else:
              right_x.extend([x_1,x_2])
              right_y.extend([y_1,y_2])
        Min_y = cap.shape[0] * (3/ 5) 
        Max_y = cap.shape[0]
        poly_l=np.poly1d(np.polyfit(left_y,left_x,deg=1))
        l_x_start = int(poly_l(Max_y))
        l_x_end = int(poly_l(Min_y))
        poly_r=np.poly1d(np.polyfit(right_y,right_x,deg=1))
        r_x_start = int(poly_r(Max_y))
        r_x_end = int(poly_r(Min_y))
        cv2.line(cap,(l_x_start,int(Max_y)),(l_x_end,int(Min_y)),(0,255,0),5)
        cv2.line(cap,(r_x_start,int(Max_y)),(r_x_end,int(Min_y)),(0,255,0),5)
        cv2.line(cap,(int(width/2),int(height)),(int(width/2),int(height-220)),(100,0,255),2)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(cap, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          cap,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=3)

      cv2.imshow('object detection', cv2.resize(cap, (800,500)))
      out.write(cap)
      if cv2.waitKey(25) & 0xFF == ord('q'):
        out.release()
        cv2.destroyAllWindows()
        break