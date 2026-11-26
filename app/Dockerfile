FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir "streamlit>=1.37,<2" "pandas>=2.2" "plotly>=5.20" "requests>=2.31"
COPY app/ app/
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
