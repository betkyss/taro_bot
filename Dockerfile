# Базовый образ
FROM python:3.9-slim

# Перемещаемся в рабочую директорию /app
WORKDIR /app

# Скопируем requirements.txt и установим зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скопируем файлы бота и т.д.
COPY bot.py .
COPY filter.png .
COPY templates ./templates

# По умолчанию команда на запуск - можно переопределить в docker-compose
CMD ["python", "bot.py"]
