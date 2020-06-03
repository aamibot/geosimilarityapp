FROM python:3.7 as python-base

COPY requirements.txt .

RUN pip install -r requirements.txt

FROM python:3.7

COPY --from=python-base /root/.cache /root/.cache

COPY --from=python-base requirements.txt .

RUN pip install -r requirements.txt && rm -rf /root/.cache

EXPOSE 8501

WORKDIR /app

COPY . .

ENTRYPOINT ["streamlit","run"]

CMD ["/app/app.py"]
