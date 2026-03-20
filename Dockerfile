FROM python:3.12.6
COPY . .
COPY requirements/app.txt /requirements.txt
RUN pip install -r requirements.txt
