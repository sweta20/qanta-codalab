FROM continuumio/anaconda3:5.3.0

RUN apt update
RUN apt install -y vim
RUN pip install awscli

COPY environment.yaml /
RUN conda env update -f environment.yaml

RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt

# RUN [ "python", "-c", "import nltk; nltk.download('all');" ]
RUN [ "python", "-m", "spacy", "download", "xx_ent_wiki_sm" ]

RUN mkdir /src
RUN mkdir /src/data
WORKDIR /src