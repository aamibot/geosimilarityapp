FROM python:3.7-slim-buster

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip/*

EXPOSE 8501

WORKDIR /app

COPY . .

ENTRYPOINT ["streamlit","run"]

CMD ["/app/app.py"]
