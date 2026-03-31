#!/usr/bin/env bash
# ============================================================
# DOCKER MAIN CONCEPTS
# ============================================================
# Docker packages applications into containers — lightweight,
# portable, self-contained units that include the code, runtime,
# libraries, and config needed to run.
#
# Key objects:
#   Image      — read-only template (blueprint)
#   Container  — running instance of an image
#   Volume     — persistent storage
#   Network    — communication between containers
#   Registry   — storage for images (e.g. Docker Hub)

# ============================================================
# 1. INSTALLATION CHECK
# ============================================================

docker --version
docker info                        # system-wide info
docker help                        # list all commands


# ============================================================
# 2. IMAGES
# ============================================================
# Images are built in layers. Each instruction in a Dockerfile
# adds a layer. Layers are cached and reused.

# --- Pull images from Docker Hub ---
docker pull ubuntu                 # latest tag (implicit)
docker pull python:3.12-slim       # specific tag
docker pull nginx:1.25-alpine      # alpine = minimal base

# --- List local images ---
docker images
docker image ls
docker image ls --filter "dangling=true"   # untagged images

# --- Inspect an image ---
docker image inspect python:3.12-slim
docker history python:3.12-slim            # show layers

# --- Remove images ---
docker rmi python:3.12-slim
docker image rm nginx:1.25-alpine
docker image prune                         # remove dangling images
docker image prune -a                      # remove ALL unused images


# ============================================================
# 3. DOCKERFILE
# ============================================================
# A Dockerfile is a text file of instructions that builds an image.
# Each instruction creates a new layer.

# Example Dockerfile (save as ./Dockerfile):
cat > Dockerfile << 'EOF'
# Base image
FROM python:3.12-slim

# Metadata
LABEL maintainer="you@example.com"
LABEL version="1.0"

# Set working directory inside the container
WORKDIR /app

# Copy dependency file first (maximizes layer cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Set environment variable
ENV APP_ENV=production
ENV PORT=8000

# Expose port (documentation only — does not publish)
EXPOSE 8000

# Default command to run
CMD ["python", "app.py"]
EOF

# Common Dockerfile instructions:
#   FROM        base image
#   WORKDIR     set working directory (creates if missing)
#   COPY        copy files from host to image
#   ADD         like COPY but supports URLs and tar auto-extraction
#   RUN         execute command during build (creates a layer)
#   ENV         set environment variable
#   ARG         build-time variable (not available at runtime)
#   EXPOSE      document which port the container listens on
#   CMD         default command (overridable at runtime)
#   ENTRYPOINT  fixed command; CMD provides default arguments
#   VOLUME      declare mount point for persistent storage
#   USER        switch to non-root user (security best practice)
#   HEALTHCHECK define a health check command


# ============================================================
# 4. BUILDING IMAGES
# ============================================================

# Build from Dockerfile in current directory
docker build -t my-app:1.0 .

# Specify Dockerfile location
docker build -f docker/Dockerfile.prod -t my-app:prod .

# Build with build arguments
docker build --build-arg APP_VERSION=1.2 -t my-app:1.2 .

# No cache (force full rebuild)
docker build --no-cache -t my-app:latest .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t my-app:latest .

# Tag an existing image
docker tag my-app:1.0 my-app:latest
docker tag my-app:1.0 myusername/my-app:1.0   # for push to Hub


# ============================================================
# 5. RUNNING CONTAINERS
# ============================================================

# Basic run
docker run ubuntu                              # run and exit
docker run -it ubuntu bash                     # interactive terminal

# Common flags
docker run \
  --name my-container \                        # name the container
  -d \                                         # detached (background)
  -p 8080:8000 \                               # host_port:container_port
  -e DATABASE_URL=postgres://... \             # environment variable
  -v /host/path:/container/path \             # bind mount
  --rm \                                       # auto-remove when stopped
  my-app:1.0

# Run a one-off command in a new container
docker run --rm python:3.12-slim python -c "print('hello')"

# Override the default command
docker run my-app:1.0 python manage.py migrate


# ============================================================
# 6. MANAGING CONTAINERS
# ============================================================

# List containers
docker ps                          # running only
docker ps -a                       # all (including stopped)
docker ps -q                       # IDs only

# Start / stop / restart
docker start my-container
docker stop my-container           # graceful (SIGTERM + wait)
docker kill my-container           # immediate (SIGKILL)
docker restart my-container

# Remove containers
docker rm my-container
docker rm -f my-container          # force remove running container
docker container prune             # remove all stopped containers

# Execute command in running container
docker exec -it my-container bash          # open shell
docker exec my-container ls /app           # run command non-interactively
docker exec -e DEBUG=1 my-container python app.py   # with env var

# Copy files between host and container
docker cp ./config.yaml my-container:/app/config.yaml
docker cp my-container:/app/logs ./logs

# Rename a container
docker rename old-name new-name


# ============================================================
# 7. LOGS & MONITORING
# ============================================================

docker logs my-container                   # print all logs
docker logs -f my-container               # follow (stream) logs
docker logs --tail 50 my-container        # last 50 lines
docker logs --since 1h my-container       # logs from last hour

# Resource usage
docker stats                               # live CPU/RAM/NET for all
docker stats my-container                  # specific container

# Inspect container details (IP, mounts, env, etc.)
docker inspect my-container
docker inspect -f '{{.NetworkSettings.IPAddress}}' my-container

# See processes running inside container
docker top my-container

# See filesystem changes since container started
docker diff my-container


# ============================================================
# 8. VOLUMES — PERSISTENT STORAGE
# ============================================================
# Volumes outlive containers and survive restarts/removals.
# Preferred over bind mounts for production data.

# --- Named volumes ---
docker volume create my-data
docker volume ls
docker volume inspect my-data
docker volume rm my-data
docker volume prune                        # remove unused volumes

# Attach a named volume to a container
docker run -d \
  --name postgres-db \
  -v my-data:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=secret \
  postgres:16

# --- Bind mounts (map host path directly) ---
docker run -d \
  -v $(pwd)/data:/app/data \              # absolute host path required
  my-app:1.0

# --- tmpfs mount (in-memory, not persisted) ---
docker run --tmpfs /tmp my-app:1.0

# Volume types summary:
#   -v my-volume:/path        named volume  (managed by Docker)
#   -v /host/path:/path       bind mount    (host filesystem)
#   --tmpfs /path             tmpfs mount   (RAM, ephemeral)


# ============================================================
# 9. NETWORKS
# ============================================================
# Containers on the same network can reach each other by name.

# --- Built-in network drivers ---
#   bridge   default for standalone containers
#   host     shares host network stack (no isolation)
#   none     no networking
#   overlay  multi-host (Docker Swarm / Compose)

# Create a custom bridge network
docker network create my-network
docker network ls
docker network inspect my-network
docker network rm my-network

# Connect containers to a network
docker run -d --name app    --network my-network my-app:1.0
docker run -d --name db     --network my-network postgres:16

# app can reach db at hostname "db" (DNS resolution by container name)

# Connect / disconnect a running container
docker network connect    my-network my-container
docker network disconnect my-network my-container


# ============================================================
# 10. DOCKER COMPOSE
# ============================================================
# Compose defines and runs multi-container apps with a YAML file.

cat > docker-compose.yml << 'EOF'
version: "3.9"

services:
  app:
    build: .                          # build from Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - REDIS_URL=redis://cache:6379
    volumes:
      - .:/app                        # bind mount for dev (live reload)
    depends_on:
      db:
        condition: service_healthy    # wait for DB health check
      cache:
        condition: service_started
    restart: unless-stopped

  db:
    image: postgres:16
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      retries: 5

  cache:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
EOF

# --- Compose commands ---
docker compose up                      # start all services (foreground)
docker compose up -d                   # detached
docker compose up --build              # rebuild images before starting
docker compose down                    # stop and remove containers
docker compose down -v                 # also remove volumes

docker compose ps                      # list service containers
docker compose logs -f app             # stream logs for a service
docker compose exec app bash           # shell into a service
docker compose run --rm app pytest     # one-off command in a new container

docker compose stop                    # stop without removing
docker compose start                   # start stopped services
docker compose restart app             # restart one service
docker compose pull                    # pull latest images

docker compose config                  # validate and view merged config


# ============================================================
# 11. REGISTRIES — PUSH & PULL
# ============================================================

# --- Docker Hub ---
docker login
docker logout

docker tag my-app:1.0 myusername/my-app:1.0
docker push myusername/my-app:1.0
docker pull myusername/my-app:1.0

# --- Private registry ---
docker login registry.example.com
docker tag my-app:1.0 registry.example.com/my-app:1.0
docker push registry.example.com/my-app:1.0

# --- GitHub Container Registry ---
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
docker tag my-app:1.0 ghcr.io/username/my-app:1.0
docker push ghcr.io/username/my-app:1.0

# Save/load image as tar (offline transfer)
docker save my-app:1.0 -o my-app.tar
docker load -i my-app.tar


# ============================================================
# 12. MULTI-STAGE BUILDS
# ============================================================
# Use multiple FROM statements to reduce final image size.
# Only the artifacts from the last stage end up in the image.

cat > Dockerfile.multistage << 'EOF'
# ---- Stage 1: build ----
FROM python:3.12 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: final (slim) ----
FROM python:3.12-slim
WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /install /usr/local
COPY . .

USER nobody                         # run as non-root
EXPOSE 8000
CMD ["python", "app.py"]
EOF

docker build -f Dockerfile.multistage -t my-app:slim .

# Benefits:
#   - Build tools, compilers, test deps stay in builder stage
#   - Final image contains only what's needed to run
#   - Significantly smaller and more secure images


# ============================================================
# 13. ENVIRONMENT VARIABLES & SECRETS
# ============================================================

# Pass env vars at runtime
docker run -e SECRET_KEY=abc123 my-app:1.0

# Load from .env file
docker run --env-file .env my-app:1.0

# In Compose, reference from shell or .env file
cat > .env << 'EOF'
POSTGRES_PASSWORD=supersecret
SECRET_KEY=change-me
DEBUG=false
EOF

# Docker secrets (Swarm / Compose v3)
cat > docker-compose.secrets.yml << 'EOF'
version: "3.9"
services:
  app:
    image: my-app:1.0
    secrets:
      - db_password
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt   # or use external: true for Swarm
EOF

# Best practices:
#   - Never bake secrets into images (they persist in layers)
#   - Use .env files locally, secret managers in production
#   - Add .env to .dockerignore


# ============================================================
# 14. HEALTH CHECKS
# ============================================================
# Docker can periodically test whether a container is healthy.

# In Dockerfile
cat >> Dockerfile << 'EOF'
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
EOF

# At runtime
docker run \
  --health-cmd="curl -f http://localhost:8000/health || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  my-app:1.0

# Check health status
docker inspect --format='{{.State.Health.Status}}' my-container

# States: starting | healthy | unhealthy


# ============================================================
# 15. .DOCKERIGNORE
# ============================================================
# Excludes files from the build context sent to the Docker daemon.
# Speeds up builds and prevents secrets leaking into images.

cat > .dockerignore << 'EOF'
# Version control
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
.venv
venv/
*.egg-info/
dist/
build/

# Tests & dev tools
.pytest_cache
.coverage
htmlcov/
.tox/

# Environment & secrets
.env
*.env
secrets/

# Editor
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Docker
Dockerfile*
docker-compose*.yml
EOF


# ============================================================
# 16. USEFUL SHORTCUTS & HOUSEKEEPING
# ============================================================

# --- System cleanup ---
docker system df                           # disk usage overview
docker system prune                        # remove stopped containers,
                                           # dangling images, unused networks
docker system prune -a                     # also remove unused images
docker system prune -a --volumes           # also remove unused volumes

# --- Bulk remove ---
docker rm $(docker ps -aq)                 # remove all stopped containers
docker rmi $(docker images -q)             # remove all images
docker stop $(docker ps -q)               # stop all running containers

# --- Format output with Go templates ---
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}"

# --- Useful run shortcuts ---
docker run --rm -it python:3.12-slim python        # ephemeral REPL
docker run --rm -v $(pwd):/data alpine ls /data    # inspect local files
docker run --rm -p 5432:5432 -e POSTGRES_PASSWORD=secret postgres:16

# --- Copy image from one tag to another without re-pulling ---
docker tag source:tag dest:tag


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# IMAGES
#   docker pull image:tag                Pull from registry
#   docker build -t name:tag .           Build from Dockerfile
#   docker images                        List local images
#   docker rmi image:tag                 Remove image
#   docker image prune -a                Remove unused images
#   docker tag src:tag dst:tag           Tag an image
#   docker push / docker pull            Push/pull to registry
#   docker save / docker load            Export/import as tar
#
# CONTAINERS
#   docker run -d -p host:cont --name    Run detached
#   docker run -it image bash            Interactive shell
#   docker run --rm image cmd            One-off, auto-remove
#   docker ps / docker ps -a             List running / all
#   docker stop / docker kill            Graceful / immediate stop
#   docker rm / docker container prune   Remove containers
#   docker exec -it name bash            Shell into running
#   docker logs -f name                  Stream logs
#   docker stats                         Live resource usage
#   docker inspect name                  Full JSON details
#   docker cp src dest                   Copy files
#
# VOLUMES
#   docker volume create / ls / rm       Manage volumes
#   -v name:/path                        Named volume
#   -v /host:/container                  Bind mount
#   --tmpfs /path                        RAM mount
#
# NETWORKS
#   docker network create / ls / rm      Manage networks
#   --network name                       Attach to network
#   Containers on same network resolve each other by name
#
# COMPOSE
#   docker compose up -d                 Start all services
#   docker compose down -v               Stop + remove volumes
#   docker compose logs -f svc           Stream service logs
#   docker compose exec svc bash         Shell into service
#   docker compose run --rm svc cmd      One-off command
#   docker compose build                 Rebuild images
#
# REGISTRY
#   docker login / docker logout         Authenticate
#   docker push user/image:tag           Upload image
#
# HOUSEKEEPING
#   docker system df                     Disk usage
#   docker system prune -a --volumes     Full cleanup
