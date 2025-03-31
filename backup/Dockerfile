FROM ubuntu:latest
LABEL authors="ajsharma"

ENTRYPOINT ["top", "-b"]

WORKDIR /workspace
ADD .. /workspace

RUN apt-get update && apt-get install -y ffmpeg libsndfile1-dev curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv pip install -e .

CMD [ "streamlit" , "run" , "/workspace/src/speech2text/main.py", "--server.address=0.0.0.0" ]

RUN mkdir /data ; chown -R 42420:42420 /workspace /data

ENV HOME=/workspace