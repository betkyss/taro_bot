# Telegram Бот для вставки фото в шаблоны

Этот проект представляет собой Telegram-бота, который принимает фотографии от пользователей, вставляет их в заранее подготовленные шаблоны и отправляет результат обратно.

## Возможности
- Выбор персонажа и шаблона (на основе имен файлов в формате `PERSONA_action.ext`)
- Вставка фото пользователя в шаблон с применением фильтра и эффектов
- Запуск через Docker и Docker Compose

## Требования
- Docker (>= 20.10)
- docker-compose-plugin
- Git

## Установка и запуск

### 1. Установка зависимостей
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git nano python3 python3-pip ca-certificates curl gnupg

# Установка Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Добавление пользователя в группу Docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Клонирование репозитория
```bash
git clone https://github.com/betkyss/taro_bot.git
cd taro_bot
```

### 3. Настройка окружения

Создайте файл `.env`:
```bash
nano .env
```

Вставьте:
```bash
BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
```
Сохраните файл.

### 4. Подготовка шаблонов

Положите файлы шаблонов в папку `templates/` с именами вида:

```
templates/
├── IREN_aura.jpg
├── IREN_finance.png
├── JUL_love.jpg
└── ...
```

Файл `filter.png` должен находиться в корне проекта.

### 5. Запуск бота через Docker Compose
```bash
docker compose build
docker compose up -d
```

### 6. Проверка работы

Откройте Telegram и отправьте команду `/start` вашему боту. Следуйте инструкциям в чате.

## Структура проекта

```
.
├── bot.py              # Код Telegram-бота
├── docker-compose.yml  # Конфигурация Docker Compose
├── Dockerfile          # Инструкция для сборки Docker образа
├── requirements.txt    # Python зависимости
├── templates/          # Шаблоны изображений
├── filter.png          # Фильтр/оверлей
└── .env                # Файл переменных окружения
```

## Полезные команды

- Запустить бота:  
  ```bash
  docker compose up -d
  ```
- Остановить бота:  
  ```bash
  docker compose down
  ```
- Перезапустить бота:  
  ```bash
  docker compose restart
  ```
- Просмотр логов:  
  ```bash
  docker compose logs -f
  ```
- Обновить код и пересобрать:
  ```bash
  git pull && docker compose build && docker compose up -d
  ```
