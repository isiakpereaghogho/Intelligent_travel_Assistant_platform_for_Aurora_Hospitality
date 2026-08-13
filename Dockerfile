FROM python:3.10-slim

WORKDIR / app

COPY . .

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main.app:app", "--host", "0.0.0.0", "--port", "8000"]