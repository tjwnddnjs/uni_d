FROM python:3.10-slim
WORKDIR /app
COPY submission_packages/final_model/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY submission_packages/final_model /app/submission_packages/final_model
ENTRYPOINT ["python", "submission_packages/final_model/test.py"]
