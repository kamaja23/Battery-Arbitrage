# Streamlit UI for the ERCOT battery arbitrage backtester.
#
# Two stages so the runtime image carries no build toolchain. The PuLP CBC
# solver is a prebuilt linux binary that needs libstdc++, which is why that
# one library is installed in the runtime stage.

FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install .


FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="ERCOT battery arbitrage backtester" \
      org.opencontainers.image.description="Streamlit UI over real ERCOT settlement point prices"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ARB_CACHE_DIR=/data/cache \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app.py ./

# Run unprivileged. /data/cache is created here so a named volume mounted on
# that path inherits these ownership bits on first use.
RUN useradd --create-home --uid 10001 arb \
 && mkdir -p /data/cache \
 && chown -R arb:arb /app /data

USER arb

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import os,sys,urllib.request; \
url='http://127.0.0.1:'+os.environ.get('STREAMLIT_SERVER_PORT','8501')+'/_stcore/health'; \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app.py"]
