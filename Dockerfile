FROM python:3.11-slim

# Use the official uv binary without relying on an install script or shell PATH.
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install the locked runtime dependencies without installing the project itself.
RUN uv sync --frozen --no-install-project

# Copy the rest of the code
COPY . .

# Ensure the environment is active by default
ENV PATH="/app/.venv/bin:$PATH"

# Default command (can be overridden)
CMD ["python", "score.py"]
