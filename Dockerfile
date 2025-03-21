FROM python:3.10

LABEL maintainer="Atharva Digambar" \
    version="1.0" \
    description="Lung Cancer Detection using Deep Learning"

WORKDIR /app
COPY . /app

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
