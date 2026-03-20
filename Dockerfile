FROM python:3.12.6
COPY . .
COPY requirements/app.txt /requirements.txt
RUN pip install -r requirements.txt
RUN pip install -e
CMD uvicorn api.fast:app --reload --host 0.0.0.0 --port $PORT
