FROM python:3.10.14-slim-bookworm
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN mkdir -p /app/data
COPY data/*.csv data/*.json /app/data/
COPY *.py /app/
WORKDIR /app
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run" ]
CMD ["app.py"]
