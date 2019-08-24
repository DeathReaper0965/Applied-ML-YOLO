from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from predict_client import YoloPredictions
import base64

HOST = '0.0.0.0'

app = Flask(__name__)
CORS(app)

yolo = YoloPredictions(HOST, 9000)


@app.route('/api/yolo', methods=['POST'])
@cross_origin()
def api():
  image_b64_arr = request.args.get("image_b64_arr")
  image_name_arr = request.args.get("image_name_arr")

  image_b64_arr = [im.strip() for im in image_b64_arr.strip("][").split(",")]
  image_name_arr = [im.strip() for im in image_name_arr.strip("][").split(",")]

  if len(image_name_arr) == len(image_b64_arr):
    return {"error": "Please check if all images are labelled in the request"}

  output_data = dict()
  for counter in range(len(image_name_arr)):
    image_b64 = base64.b64decode(str(image_b64_arr[counter]).replace(" ", "+"))
    output_data[image_name_arr[counter]] = yolo.predict_image(image_b64)

  response = jsonify(output_data)
  return response


if __name__ == '__main__':
    app.run(host=HOST, debug=True, port=8501)
