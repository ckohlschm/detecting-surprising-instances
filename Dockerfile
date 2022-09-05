FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN apt-get update && apt-get -y upgrade && apt -y install graphviz graphviz-dev xdg-utils r-base
RUN R -e "install.packages('BiocManager')"
RUN R -e "BiocManager::install(c('igraph'))"
RUN R -e "BiocManager::install(c('SID', 'bnlearn', 'pcalg', 'kpcalg', 'glmnet', 'mboost'))"
RUN R -e "BiocManager::install(c('ccdrAlgorithm', 'discretecdAlgorithm'))"
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN [ "python", "./frontend/manage.py", "migrate" ]
CMD [ "sh", "./scripts/startup.sh" ]

EXPOSE 33333