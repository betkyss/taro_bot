#!/bin/bash

# Функция для выполнения команды и отображения её логов
run_command() {
  echo "Выполнение команды: $*"
  "$@" || {
    echo "Ошибка выполнения команды: $*" >&2
    exit 1
  }
}

# Остановка текущих контейнеров Docker
stop_docker_compose() {
  echo "Остановка текущих контейнеров Docker..."
  run_command docker-compose down
}

# Проверка существующих образов Docker
check_existing_images() {
  echo "Проверка существующих образов Docker..."
  local images_exist=true

  # Проверяем, существуют ли образы для всех сервисов в docker-compose.yml
  while IFS= read -r service; do
    if ! docker images "$service" | grep -q "$service"; then
      echo "Образ для $service не найден, требуется сборка."
      images_exist=false
    fi
  done < <(docker-compose config | awk '/image:/{print $2}')

  echo "Образы найдены: $images_exist"
  $images_exist
}

# Пересборка образа Docker, если это необходимо
build_docker_image_if_needed() {
  echo "Пересборка образов Docker, если это необходимо..."
  if check_existing_images; then
    echo "Образы уже существуют, пересборка не требуется."
  else
    echo "Образы отсутствуют или устарели, выполняется сборка..."
    run_command docker-compose build
  fi
}

# Запуск контейнеров Docker
start_docker_compose() {
  echo "Запуск контейнеров Docker..."
  run_command docker-compose up -d
}

# Главная часть
echo "Начало процесса управления Docker..."
stop_docker_compose
build_docker_image_if_needed
start_docker_compose
echo "Все процессы завершены успешно!"
