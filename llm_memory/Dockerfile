FROM python:3.11-bookworm as builder

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

FROM python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY --from=builder /venv /venv
COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE 8086

ENTRYPOINT [ "/venv/bin/python" ]
CMD [ "rest_server.py" ]