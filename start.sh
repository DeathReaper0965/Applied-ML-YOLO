cd && git clone https://github.com/DeathReaper0965/aml_yolo_backend_task

git clone https://github.com/tensorflow/serving

cd serving/tensorflow_serving/tools/docker

docker build --pull -t $USER/tensorflow-serving-devel-cpu -f Dockerfile.devel .

docker run --name yolo_contianer -it -d -p 8501:8501 $USER/tensorflow-serving-devel-cpu /bin/bash

cd && docker cp aml_yolo_backend_task yolo_contianer:/tensorflow-serving/

docker exec -it yolo_contianer /bin/bash

tensorflow_model_server --port=9000 --model_name=0 --model_base_path=/tensorflow-serving/aml_yolo_backend_task/models

pip install flask gunicorn flask_cors tensorflow keras opencv-python numpy tensorflow-serving-api

cd /tensorflow-serving/aml_yolo_backend_task/

gunicorn -b 0.0.0.0:8501 wsgi:app -w 3 -m 007 --limit-request-line 0
