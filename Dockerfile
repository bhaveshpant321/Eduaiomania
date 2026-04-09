# Use lightweight python base
FROM python:3.10-slim

WORKDIR /app

# Install dependencies via the project file
COPY pyproject.toml .
COPY README.md .
# Create the package directory and copy content
COPY eudaimonia/ /app/eudaimonia/

# Install the package and its dependencies
RUN pip install --no-cache-dir .

# Set Python path
ENV PYTHONPATH=/app

# Default HF spaces port
EXPOSE 7860

# Run the backend using the package path
CMD ["uvicorn", "eudaimonia.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
