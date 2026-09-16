FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
COPY marketcanvas/ marketcanvas/
COPY demo.py interactive.py ./
COPY templates/ templates/

RUN pip install --no-cache-dir -e ".[dev]" flask

EXPOSE 8080

CMD ["python", "demo.py"]
