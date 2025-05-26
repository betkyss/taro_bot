# Базовый образ — компактная версия Python 3.9 на Debian Buster
FROM python:3.9-slim-buster

# Отключаем буферизацию вывода Python, чтобы логи сразу шли в stdout
ENV PYTHONUNBUFFERED=1

# Рабочая директория внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей и устанавливаем Python-библиотеки
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код бота и вспомогательные файлы
COPY bot.py .
COPY filter.png .
COPY templates ./templates
COPY allowed_users.txt .

# Запускаем бота
CMD ["python", "bot.py"]

