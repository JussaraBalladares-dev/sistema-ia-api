FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y app
#./api es carpeta que esta al mismo nivel que docker-compose
COPY ./api/requirements.txt .
COPY ./api/ ./app

# Instalar dependencias bÃ¡sicas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python (usado desde el docker-compose)
