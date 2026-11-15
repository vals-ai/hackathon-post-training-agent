FROM python:3.12

RUN pip install --no-cache-dir uv==0.13.5

WORKDIR /opt/post_training_agent
COPY pyproject.toml uv.lock README.md ./
COPY src src
RUN uv sync --frozen --no-dev
COPY run.sh ./
RUN chmod +x run.sh

RUN install -d -o root -g root -m 0700 /logs && install -d -o root -g root -m 0755 /workspace
