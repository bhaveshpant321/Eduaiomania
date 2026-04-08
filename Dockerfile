# Use lightweight python base
FROM python:3.10-slim

WORKDIR /app

# Copy requirements or just install directly
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Add the project files
COPY engine/ /app/engine/
COPY server/ /app/server/

# Set Python path
ENV PYTHONPATH=/app

# Default HF spaces port
EXPOSE 7860

# Run the backend
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
