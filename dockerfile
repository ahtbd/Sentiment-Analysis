FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --timeout 600 -r requirements.txt

COPY app ./app

EXPOSE 7860

CMD ["python", "-m", "app.ui"]
