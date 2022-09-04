FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN apt-get update && apt-get -y upgrade && apt -y install graphviz graphviz-dev xdg-utils r-base
RUN R -e "install.packages(c('pcalg', 'kpcalg', 'bnlearn', 'sparsebn', 'SID', 'CAM', 'D2C', 'RCIT'), dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN [ "python", "./frontend/manage.py", "migrate" ]
CMD [ "sh", "./scripts/startup.sh" ]

EXPOSE 33333