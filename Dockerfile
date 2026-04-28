FROM python:3.10-slim

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    FINAL_JSON=outputs/test_single/final_pipeline_output.json \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY outputs ./outputs
COPY prompts ./prompts
COPY configs ./configs

RUN pip install --no-cache-dir --no-build-isolation -e ".[deploy-mysql]"

EXPOSE 7860

CMD ["sh", "-c", "python outputs/display_final_pipeline_gradio.py --final-json \"$FINAL_JSON\" --port \"$GRADIO_SERVER_PORT\""]
