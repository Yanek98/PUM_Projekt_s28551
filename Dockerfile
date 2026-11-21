FROM apache/airflow:2.11.0-python3.11
COPY requirements.txt /tmp/steam-requirements.txt
RUN pip install --no-cache-dir "apache-airflow==2.11.0" -r /tmp/steam-requirements.txt
