FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# docker build -t reqlens .
# docker run --rm --env-file .env -v "$(pwd)/reports:/app/reports" reqlens examples/requirements.md reports/report.md
ENTRYPOINT ["python", "main.py"]
CMD ["examples/requirements.md", "reports/report.md"]
