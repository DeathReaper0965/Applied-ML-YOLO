from grpc.beta import implementations
import numpy as np
import cv2 as cv
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto


class YoloPredictions:

  def __init__(self, host, port):
    self.inpWidth, self.inpHeight = 416, 416
    self.timeout = 60.0
    channel = implementations.insecure_channel(host, int(port))
    self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  def predict_image(self, image_b64):
    nparr = np.fromstring(image_b64, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_ANYCOLOR)
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)

    request = predict_pb2.PredictRequest()

    request.model_spec.name = '0'
    request.model_spec.signature_name = 'predict_image'
    request.inputs['image'].CopyFrom(make_tensor_proto(blob, shape=list(blob.shape)))

    # print("Going to predict")
    response = self.stub.Predict(request, self.timeout)
    # print(f"Got response: {response}")
    result = response.outputs['scores']

    imageHeight = image.shape[0]
    imageWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    for out in result:
      for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > 0.1:
          center_x = int(detection[0] * imageWidth)
          center_y = int(detection[1] * imageHeight)
          width = int(detection[2] * imageWidth)
          height = int(detection[3] * imageHeight)
          left = int(center_x - width / 2)
          top = int(center_y - height / 2)
          classIds.append(classId)
          confidences.append(float(confidence))
          boxes.append([left, top, width, height])

    final_response = []
    for resp_count in range(len(classIds)):
      final_response.append({"class": classIds[resp_count],
                             "confidence": confidences[resp_count],
                             "boxes": boxes[resp_count]})

    return final_response
