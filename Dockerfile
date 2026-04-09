# Use lightweight python base
FROM python:3.10-slim

WORKDIR /app

# Copy requirements or just install directly
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai openenv-core

# Add the project files (back to root to satisfy validator)
COPY engine/ /app/engine/
COPY server/ /app/server/
COPY README.md /app/
COPY openenv.yaml /app/

# Set Python path
ENV PYTHONPATH=/app

# Default HF spaces port
EXPOSE 7860

# Run the backend using the standard path
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
