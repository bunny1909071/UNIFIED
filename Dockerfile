FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Fonts + libs for WeasyPrint. fonts-noto-core includes Telugu, Devanagari, Tamil,
# Kannada, Malayalam and Bengali, so no runtime apt-get / GitHub downloads are needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libharfbuzz-subset0 fontconfig fonts-noto-core fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* && fc-cache -f

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE 7860
# One gthread worker with 8 threads: the app mostly waits on AI APIs, and a single process keeps the
# rate limits, result cache and Gemini cooldown consistent (two workers each kept their own copy).
# timeout > slowest AI call chain (vision 120s + fallback).
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", "--timeout", "180", "--preload", "app:app"]
