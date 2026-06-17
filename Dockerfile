FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    OPENREVIEW_PIPELINE_LOG_DIR=/app/outputs/logs \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    FINAL_JSON=outputs/final_pipeline_output.json

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY outputs ./outputs
COPY configs ./configs
COPY tests/scripts ./tests/scripts

RUN pip install --no-cache-dir -U pip \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -e ".[deploy-postgres]"

EXPOSE 7860

CMD ["sh", "-c", "python outputs/display_final_pipeline_gradio.py --final-json \"$FINAL_JSON\" --port \"$GRADIO_SERVER_PORT\""]
