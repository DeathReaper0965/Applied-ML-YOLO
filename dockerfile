FROM python:3.6
ADD . /app
WORKDIR /app
RUN pip install flask gunicorn flask_cors tensorflow keras opencv-python numpy tensorflow-serving-api
EXPOSE 8501
CMD ["gunicorn", "-b", "0.0.0.0:8501", "app", "-w", "10", "--limit-request-line", "0"]