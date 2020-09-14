## Applied Machine Learning for YOLO Object Detection

This is a custom YOLO object detection model trained using tiny-yolov3 in order to re-train the feature layers for detecting 3 classes namely: Arms, Legs and Hair of the people in a given Image. 

The Application is backed by **Supervisor, Gunicorn and WSGI servers** for handling Multiple API requests parellelly. As the model is deployed using TensorFlow Serving API on Docker, the load balancing of requests can be handled much more robustly.

#### Steps to be followed in order to make a successful prediction request: 

1. Install required libraries using **pip install -r requirements.txt**
2. For Local Deployment: Directly run the flask app using the command **python app.py**
3. For Docker Deployments please check the included **dockerfile**
4. Once the server is up and running you can make a request by pass in images as an array of **base64 strings** along with their respective **image names**
5. You can check the file **conc_requests.py** file which bombards 100000 requests which can all be served without any loss when deployed on Docker with 10 Gunicorn workers.

Below is the sample of the input image and the output bounding boxes displayed over the image.

![Original Image](https://github.com/DeathReaper0965/Applied-ML-YOLO/blob/master/images/test.jpg?raw=true) ![Predicted Image](https://github.com/DeathReaper0965/Applied-ML-YOLO/blob/master/images/required_output.jpg?raw=true)

Made with ❤️ &nbsp;by Praneet Pabolu
