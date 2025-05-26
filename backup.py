import os
import io
import math
import random
import time
import traceback
import numpy as np

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, Chat, User

############################################
# 1) ЗАГРУЗКА БЕЛОГО СПИСКА ИЗ ФАЙЛА allowed_users.txt
############################################
def load_allowed_user_ids(filename="allowed_users.txt"):
    """Читает allowed_users.txt и возвращает множество user_id (целых чисел)."""
    allowed = set()
    if not os.path.exists(filename):
        print(f"[DEBUG] Файл {filename} не найден. Белый список пуст.")
        return allowed
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        allowed.add(int(line))
                    except ValueError:
                        print(f"[DEBUG] Невалидный user_id в файле {filename}: '{line}'")
        print(f"[DEBUG] Загружено {len(allowed)} user_id из {filename}: {allowed}")
    except Exception as e:
        print("[DEBUG] Ошибка при загрузке белого списка:", e)
    return allowed

ALLOWED_USER_IDS = load_allowed_user_ids("allowed_users.txt")

############################################
# 2) ЗАГРУЗКА ТОКЕНА И НАСТРОЙКИ БОТА
############################################
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Увеличиваем таймауты для отправки файлов
import telebot.apihelper
telebot.apihelper.SEND_FILE_TIMEOUT = 120
telebot.apihelper.CONNECT_TIMEOUT = 30
telebot.apihelper.READ_TIMEOUT = 30

bot = telebot.TeleBot(BOT_TOKEN)
print("Бот включён и готов к работе.")

templates_dir = 'templates'
filter_path = 'filter.png'

# Параметры (обновлены в соответствии со вторым примером)
min_shift = 5
max_shift = 15
min_rotation = 2
max_rotation = 4
scale_pixels = 35
upscale_factor = 5.0
thickness = 25
box_blur_radius = 5
MAX_TELEGRAM_DIM = 4096

user_data = {}  # Хранилище данных (персона, действие, состояние)

############################################
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
############################################
def list_templates():
    """Возвращает список файлов шаблонов (.png, .jpg, .jpeg) в папке templates."""
    files = [f for f in os.listdir(templates_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[DEBUG] list_templates: Найдено {len(files)} файлов -> {files}")
    return files

def build_persona_actions_mapping():
    """
    Строит словарь вида:
      { 'IREN': {'aura': 'IREN_aura.jpeg', ...}, ... }
    на основе имён файлов вида PERSONA_action.jpg/png/jpeg.
    """
    mapping = {}
    for filename in list_templates():
        name, _ = os.path.splitext(filename)
        parts = name.split('_')
        if len(parts) < 2:
            continue
        persona = parts[0]
        action = parts[1]
        if persona not in mapping:
            mapping[persona] = {}
        mapping[persona][action] = filename
    print(f"[DEBUG] build_persona_actions_mapping -> {mapping}")
    return mapping

############################################
# 3) ОСНОВНАЯ ФУНКЦИЯ ВСТАВКИ ФОТО
############################################
def process_template_photo(template_img: Image.Image, user_photo_img: Image.Image) -> bytes:
    """
    Логика вставки из второго примера.
    Возвращает итоговое изображение как байты (JPEG).
    """
    w, h = template_img.size
    arr = np.array(template_img)
    # Ищем зелёный цвет в шаблоне
    green = (arr[:, :, 0] < 100) & (arr[:, :, 1] > 200) & (arr[:, :, 2] < 100)
    mask_orig = Image.fromarray((green * 255).astype(np.uint8), mode='L')
    bbox = mask_orig.getbbox()
    if not bbox:
        # Если нет зелёной области, просто возвращаем исходный шаблон (RGB)
        buf = io.BytesIO()
        template_img.convert('RGB').save(buf, format='JPEG', quality=90)
        buf.seek(0)
        return buf.getvalue()

    # Центр зелёной области
    cx_orig = (bbox[0] + bbox[2]) // 2
    cy_orig = (bbox[1] + bbox[3]) // 2

    # Апскейл шаблона
    up_w, up_h = int(w * upscale_factor), int(h * upscale_factor)
    template_up = template_img.resize((up_w, up_h), Image.BICUBIC).convert('RGBA')
    cx, cy = int(cx_orig * upscale_factor), int(cy_orig * upscale_factor)

    bbox_up = (
        int(bbox[0] * upscale_factor), int(bbox[1] * upscale_factor),
        int(bbox[2] * upscale_factor), int(bbox[3] * upscale_factor)
    )
    base_w = bbox_up[2] - bbox_up[0]
    base_h = bbox_up[3] - bbox_up[1]
    scale_pixels_up = int(scale_pixels * upscale_factor)
    crop_w = base_w + scale_pixels_up
    crop_h = base_h + scale_pixels_up

    # Апскейл пользовательского фото
    user_photo_img = user_photo_img.convert('RGBA')
    p_w, p_h = user_photo_img.size
    photo_up = user_photo_img.resize((int(p_w * upscale_factor), int(p_h * upscale_factor)), Image.BICUBIC)
    pu_w, pu_h = photo_up.size

    # Масштабируем так, чтобы фото перекрывало зелёную область с небольшим запасом
    sf = max(crop_w / pu_w, crop_h / pu_h)
    photo_scaled = photo_up.resize((int(pu_w * sf), int(pu_h * sf)), Image.BICUBIC)
    ps_w, ps_h = photo_scaled.size

    crop_left = (ps_w - crop_w) // 2
    crop_top = (ps_h - crop_h) // 2
    cropped_photo = photo_scaled.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))

    # Случайный поворот и сдвиг
    angle = random.choice([-1, 1]) * random.randint(min_rotation, max_rotation)
    dx = random.randint(min_shift, max_shift) * random.choice([-1, 1])
    dy = random.randint(min_shift, max_shift) * random.choice([-1, 1])
    rotated_photo = cropped_photo.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0,0,0,0))

    target_center = (cx + dx, cy + dy)
    paste_x = target_center[0] - (rotated_photo.width // 2)
    paste_y = target_center[1] - (rotated_photo.height // 2)

    # Наклейка/фильтр
    filter_img = Image.open(filter_path).convert('RGBA')
    filter_resized = filter_img.resize((cropped_photo.width, cropped_photo.height + 30), Image.BICUBIC)
    rotated_filter = filter_resized.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0,0,0,0))

    # Склеиваем фото и фильтр
    composite = Image.new('RGBA', rotated_filter.size, (0,0,0,0))
    composite.paste(rotated_photo, (0,0))
    final_composite = Image.alpha_composite(composite, rotated_filter)

    # Вклеиваем результат в шаблон
    layer = Image.new('RGBA', template_up.size, (0,0,0,0))
    layer.paste(final_composite, (paste_x, paste_y))
    final_img = Image.alpha_composite(template_up, layer)

    # Размываем левый край
    corners = [(0, 0), (crop_w, 0), (crop_w, crop_h), (0, crop_h)]
    cx_crop, cy_crop = crop_w / 2, crop_h / 2
    rotated_corners = []
    rad = math.radians(-angle)
    for x, y in corners:
        dx_local, dy_local = x - cx_crop, y - cy_crop
        rx = dx_local * math.cos(rad) - dy_local * math.sin(rad)
        ry = dx_local * math.sin(rad) + dy_local * math.cos(rad)
        rx += rotated_photo.width / 2 + paste_x
        ry += rotated_photo.height / 2 + paste_y
        rotated_corners.append((rx, ry))

    # p1, p2 – это та сторона, которую хотим слегка размыть
    p1, p2 = rotated_corners[0], rotated_corners[3]
    dx_edge, dy_edge = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx_edge, dy_edge)
    if length != 0:
        nx, ny = (-dy_edge / length, dx_edge / length)
        offset = thickness / 2
        polygon = [
            (p1[0] + nx*offset, p1[1] + ny*offset),
            (p2[0] + nx*offset, p2[1] + ny*offset),
            (p2[0] - nx*offset, p2[1] - ny*offset),
            (p1[0] - nx*offset, p1[1] - ny*offset)
        ]
        mask = Image.new('L', final_img.size, 0)
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        blurred = final_img.filter(ImageFilter.BoxBlur(box_blur_radius))
        final_img = Image.composite(blurred, final_img, mask)

    # Проверяем итоговые размеры
    if final_img.width > MAX_TELEGRAM_DIM or final_img.height > MAX_TELEGRAM_DIM:
        final_img.thumbnail((MAX_TELEGRAM_DIM, MAX_TELEGRAM_DIM))

    # Конвертируем в RGB и возвращаем как JPEG-байты
    final_img = final_img.convert('RGB')
    out_buf = io.BytesIO()
    final_img.save(out_buf, format='JPEG', quality=90)
    out_buf.seek(0)
    return out_buf.getvalue()

############################################
# 4) ХЕНДЛЕРЫ
############################################
@bot.message_handler(commands=['start'])
def cmd_start(message):
    """Стартовая команда. Если пользователь не в белом списке, сообщаем об отсутствии доступа."""
    if message.from_user.id not in ALLOWED_USER_IDS:
        bot.send_message(message.chat.id, "У вас нет доступа к этому боту.")
        return
    print("[DEBUG] /start команду вызвал пользователь:", message.chat.id)
    mapping = build_persona_actions_mapping()
    if not mapping:
        bot.send_message(message.chat.id, "Нет шаблонов.")
        return
    markup = InlineKeyboardMarkup()
    for persona in sorted(mapping.keys()):
        btn = InlineKeyboardButton(persona, callback_data=f"persona_{persona}")
        markup.add(btn)
    bot.send_message(message.chat.id, "🎭Выберите персонажа:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("persona_") or call.data.startswith("action_"))
def handle_persona_action_callback(call):
    """Обработка callback'ов для выбора персонажа и действия."""
    if call.from_user.id not in ALLOWED_USER_IDS:
        bot.answer_callback_query(call.id, text="У вас нет доступа к этому боту.")
        return

    if call.data.startswith("persona_"):
        persona = call.data.replace("persona_", "")
        print(f"[DEBUG] Пользователь {call.message.chat.id} выбрал персонажа: {persona}")
        mapping = build_persona_actions_mapping()
        if persona not in mapping:
            bot.answer_callback_query(call.id, text="Такого персонажа нет.")
            return
        user_data[call.message.chat.id] = {"persona": persona, "state": "choose_action"}
        bot.answer_callback_query(call.id, text=f"Выбран персонаж: {persona}")
        actions = sorted(mapping[persona].keys())
        markup = InlineKeyboardMarkup()
        for act in actions:
            markup.add(InlineKeyboardButton(act, callback_data=f"action_{act}"))
        bot.send_message(call.message.chat.id, "🖼️Выберите шаблон:", reply_markup=markup)
    elif call.data.startswith("action_"):
        action = call.data.replace("action_", "")
        chat_id = call.message.chat.id
        print(f"[DEBUG] Пользователь {chat_id} выбрал действие: {action}")
        persona = user_data.get(chat_id, {}).get("persona")
        if not persona:
            bot.answer_callback_query(call.id, text="Сначала выберите персонажа.")
            return
        user_data[chat_id]["action"] = action
        user_data[chat_id]["state"] = "waiting_photo"
        bot.answer_callback_query(call.id, text=f"Выбрано: {action}")
        bot.send_message(chat_id, "📥Отправь фото для вставки")

@bot.callback_query_handler(func=lambda call: call.data == "create_more")
def handle_create_more_callback(call):
    """Обработка callback'а 'create_more'. Перезапуск диалога выбора персонажа."""
    if call.from_user.id not in ALLOWED_USER_IDS:
        bot.answer_callback_query(call.id, text="У вас нет доступа к этому боту.")
        return
    print("[DEBUG] callback_create_more: пользователь хочет ещё.")
    bot.answer_callback_query(call.id)
    dummy_msg = call.message
    dummy_msg.from_user = call.from_user
    cmd_start(dummy_msg)

@bot.message_handler(content_types=['photo','document'])
def handle_photo_or_document(message):
    """Обработка фото/документа. Проверяем белый список и состояние 'waiting_photo'."""
    if message.from_user.id not in ALLOWED_USER_IDS:
        bot.send_message(message.chat.id, "У вас нет доступа к этому боту.")
        return

    chat_id = message.chat.id
    print(f"[DEBUG] handle_photo_or_document от пользователя {chat_id}. Тип: {message.content_type}")
    if user_data.get(chat_id, {}).get("state") != "waiting_photo":
        print("[DEBUG] Пользователь не в состоянии waiting_photo.")
        bot.send_message(chat_id, "Сначала выберите шаблон /start.")
        return

    bot.send_message(chat_id, "⏳Идёт обработка...")
    persona = user_data[chat_id]["persona"]
    action = user_data[chat_id]["action"]
    mapping = build_persona_actions_mapping()
    tmpl_filename = mapping[persona][action]
    template_path = os.path.join(templates_dir, tmpl_filename)
    if not os.path.exists(template_path):
        print("[DEBUG] Шаблон не найден:", template_path)
        bot.send_message(chat_id, "Шаблон не найден.")
        return
    print("[DEBUG] Используем шаблон:", template_path)

    if message.content_type == 'photo':
        file_id = message.photo[-1].file_id
    else:
        file_id = message.document.file_id
    print(f"[DEBUG] Скачиваем файл с file_id={file_id}")

    file_info = bot.get_file(file_id)
    downloaded = bot.download_file(file_info.file_path)
    user_photo_buf = io.BytesIO(downloaded)

    try:
        user_photo = Image.open(user_photo_buf)
        print(f"[DEBUG] Успешно открыли фото. Размер: {user_photo.size}")
    except Exception as e:
        print("[DEBUG] Не удалось открыть файл как изображение:", e)
        bot.send_message(chat_id, "Не удалось открыть файл как изображение.")
        return

    template_img = Image.open(template_path)
    print(f"[DEBUG] Открыли шаблон. Размер: {template_img.size}")

    try:
        result_bytes = process_template_photo(template_img, user_photo)
    except Exception as e:
        print("[DEBUG] Ошибка при процессе вставки:", e)
        traceback.print_exc()
        bot.send_message(chat_id, "Произошла ошибка при обработке изображения.")
        return

    again_markup = InlineKeyboardMarkup()
    again_markup.add(InlineKeyboardButton("🔄Сгенерировать фото", callback_data="create_more"))
    print("[DEBUG] Отправляем итоговое изображение пользователю.")
    bot.send_photo(chat_id, result_bytes, caption="✅ Готово!", reply_markup=again_markup)

    user_data[chat_id]["state"] = None
    user_photo.close()
    template_img.close()

############################################
# ЗАПУСК БОТА
############################################
if __name__ == "__main__":
    print("[DEBUG] Удаляем вебхук, если был установлен.")
    bot.remove_webhook()
    print("[DEBUG] Запуск bot.infinity_polling()")
    while True:
        try:
            bot.infinity_polling()
        except Exception as e:
            print("[FATAL ERROR]", e)
            traceback.print_exc()
            print("Перезапуск через 5 секунд...")
            time.sleep(5)

