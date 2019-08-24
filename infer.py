import cv2 as cv
import numpy as np
import os.path
from convert_model import convert_model_to_tf
import math


confThreshold = 0.1
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416


class YOLOHelper:

  def __init__(self, model_configuration_file, model_weights_file, classes_file):
    assert(os.path.exists(classes_file))
    with open(classes_file, 'rt') as f:
      self.classes = f.read().rstrip('\n').split('\n')

    self.net = self.load_model(model_configuration_file, model_weights_file)

  def load_model(self, model_configuration_file, model_weights_file):
    net = cv.dnn.readNetFromDarknet(model_configuration_file, model_weights_file)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net

  def get_outputs_names(self, net):
    layersNames = self.net.getLayerNames()
    return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

  def draw_pred(self, image, classId, conf, left, top, right, bottom):
    cv.rectangle(image, (left, top), (left + right, top + bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    assert(classId < len(self.classes))
    label = '%s:%s' % (self.classes[classId], label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(image, (left, top - int(round(1.5*labelSize[1]))), (int(left + round(1.5*labelSize[0])), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    return image

  def postprocess(self, image, outs):
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    for out in outs:
      for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if math.ceil(confidence) > confThreshold:
          center_x = int(detection[0] * imageWidth)
          center_y = int(detection[1] * imageHeight)
          width = int(detection[2] * imageWidth)
          height = int(detection[3] * imageHeight)
          left = int(center_x - (width / 2))
          top = int(center_y - (height / 2))
          classIds.append(classId)
          confidences.append(float(confidence))
          boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
      i = i[0]
      box = boxes[i]
      left = box[0]
      top = box[1]
      width = box[2]
      height = box[3]
      image = self.draw_pred(image, classIds[i], confidences[i], left, top, width, height)
    return [self.classes[v] for v in classIds], confidences, boxes, image

  def predict(self, image):
    blob = cv.dnn.blobFromImage(image, 1/255.0, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    self.net.setInput(blob)
    outs = self.net.forward(self.get_outputs_names(self.net))
    return outs

  def infer_image(self, image):
    outs = self.predict(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return self.postprocess(image, outs)


def main():
  image = cv.imread("test.jpg")
  yolo_helper = YOLOHelper("tiny_yolov3.cfg", "tiny_yolov3.weights", "tiny_yolov3.names")
  classes, confidences, boxes, image = yolo_helper.infer_image(image)
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

  to_convert = False
  if to_convert:
    trained_checkpoint_prefix = 'DW2TF/data/tiny_yolov3.ckpt'
    version = 0
    convert_model_to_tf(trained_checkpoint_prefix, version)
  cv.imshow("result", image)
  cv.waitKey()


if __name__== "__main__":
  main()