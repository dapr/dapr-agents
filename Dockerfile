FROM squidfunk/mkdocs-material

RUN apk update && \
    apk add gcc python3-dev musl-dev linux-headers && \
    pip install mkdocs-jupyter