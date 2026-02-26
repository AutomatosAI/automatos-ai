FROM python:3.12-slim

WORKDIR /app

# System tools agents need (git, node, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget openssh-client ca-certificates \
    build-essential jq tree \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g pnpm \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Worker code
COPY . .

RUN useradd -m -u 1000 worker \
    && mkdir -p /workspaces \
    && chown -R worker:worker /app /workspaces
USER worker

EXPOSE 8081

CMD ["python", "-m", "main"]
