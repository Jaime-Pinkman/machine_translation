FROM python:3.10-slim-buster

# Install Poetry
ARG POETRY_VERSION=1.2.2
ENV POETRY_VERSION=$POETRY_VERSION
RUN pip install poetry==$POETRY_VERSION

# Set working directory
WORKDIR /app

# Copy project files to container
COPY . /app

# Install project dependencies using Poetry
RUN poetry install --no-root

# Run pytest
CMD ["poetry", "run", "pytest"]
