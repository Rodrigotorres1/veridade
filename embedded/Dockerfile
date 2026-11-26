FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir "paho-mqtt>=2.0" "psycopg2-binary>=2.9" "pandas>=2.2" "pyarrow>=15" "numpy>=1.26"
COPY embedded/ embedded/
ENV PYTHONUNBUFFERED=1
