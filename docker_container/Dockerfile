FROM python:3.7
WORKDIR /app
COPY task_run.py /app
RUN pip install google-cloud-aiplatform==1.34.0 pytorch-forecasting==0.10.1 pytorch-lightning==1.9.5 torch==1.13.1
CMD ["python", "./task_run.py"]