FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cache layer)
COPY setup.py pyproject.toml ./
COPY wmcp/ wmcp/
RUN pip install --no-cache-dir .

# Copy demo and spec (optional)
COPY demo/ demo/
COPY protocol-spec/ protocol-spec/

ENTRYPOINT ["python", "-m", "wmcp.cli"]
CMD ["info"]
