FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN apt-get update && apt-get -y upgrade && apt -y install graphviz xdg-utils
RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=api.py
ENV FLASK_ENV=production

COPY . .

RUN [ "python", "./frontend/manage.py", "migrate" ]
CMD [ "sh", "./scripts/startup.sh" ]

EXPOSE 33333