FROM apache/airflow:2.11.0-python3.11
COPY requirements.txt /tmp/steam-requirements.txt
RUN pip install --no-cache-dir "apache-airflow==2.11.0" -r /tmp/steam-requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
RUN pip install --no-cache-dir "apache-airflow==2.11.0" "mlflow==2.22.2" "dask==2024.12.1" \
    "cloudpickle==3.0.0" "protobuf==4.25.7" "opentelemetry-exporter-otlp==1.27.0" \
    -r /tmp/steam-requirements.txt && python -m pip check
RUN python -c "from torchvision.models import MobileNet_V3_Small_Weights as M, ShuffleNet_V2_X0_5_Weights as S; [w.DEFAULT.get_state_dict(progress=False, check_hash=True) for w in (M, S)]"
