# Базовый образ — компактная версия Python 3.9
FROM python:3.9-slim

# Отключаем буферизацию вывода Python, чтобы логи сразу шли в stdout
ENV PYTHONUNBUFFERED=1

# Рабочая директория внутри контейнера
WORKDIR /app

# Скопируем файл зависимостей и установим их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код бота и нужные файлы (включая allowed_users.txt)
COPY bot.py .
COPY filter.png .
COPY templates ./templates
COPY allowed_users.txt . 

# Команда по умолчанию — запуск бота
CMD ["python", "bot.py"]

