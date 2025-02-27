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
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# Загружаем переменные среды (где лежит BOT_TOKEN)
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Инициализация бота
bot = telebot.TeleBot(BOT_TOKEN)
print("Бот включён и готов к работе.")

# Папка с шаблонами и путь к фильтру
templates_dir = 'templates'
filter_path = 'filter.png'

# Параметры обработки
min_shift = 5
max_shift = 15
min_rotation = 2
max_rotation = 4
scale_pixels = 35
upscale_factor = 5.0        # Здесь выставляем 5-кратное увеличение
thickness = 25
box_blur_radius = 5
MAX_TELEGRAM_DIM = 4096     # Максимальный размер для отправки в Telegram

# Хранилище данных по пользователям (выбранный персонаж, действие и т. д.)
user_data = {}

def list_templates():
    """
    Возвращает список всех файлов из папки templates,
    у которых расширение .png / .jpg / .jpeg
    """
    files = [f for f in os.listdir(templates_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[DEBUG] list_templates: Найдено {len(files)} файлов в {templates_dir} -> {files}")
    return files

def build_persona_actions_mapping():
    """
    Строит структуру вида:
    {
      'IREN': {'aura': 'IREN_aura.jpeg', 'finance': 'IREN_finance.jpeg', ...},
      'JUL': {'love': 'JUL_love.jpeg', ...}
    }
    На основании файлов, названных в формате PERSONA_action.jpeg
    """
    mapping = {}
    for filename in list_templates():
        name, _ = os.path.splitext(filename)    # Пример: IREN_aura, .jpeg
        parts = name.split('_')                 # ['IREN', 'aura']
        if len(parts) < 2:
            # Файл не соответствует формату, пропускаем
            continue
        persona = parts[0]  # IREN
        action = parts[1]   # aura
        if persona not in mapping:
            mapping[persona] = {}
        # Каждой паре (persona, action) сопоставляем имя файла
        mapping[persona][action] = filename

    print(f"[DEBUG] build_persona_actions_mapping -> {mapping}")
    return mapping

def process_template_photo(template_img: Image.Image, user_photo_img: Image.Image) -> bytes:
    """
    1) Вставляет user_photo_img в зелёную область template_img,
       используя upscale_factor = 5.0, поворот/сдвиг, фильтр, размытие.
    2) Сохраняет результат (большое изображение) во временный файл -> закрывает объект.
    3) Заново открывает файл, если размер > MAX_TELEGRAM_DIM, делает thumbnail,
       снова сохраняет в BytesIO и возвращает байты итогового JPEG.
    """
    print("[DEBUG] Начало process_template_photo...")
    w, h = template_img.size
    print(f"[DEBUG] Размер шаблона (template_img): {w}x{h}")
    arr = np.array(template_img)

    # Определяем пиксели, где "зелёный" цвет
    green = (arr[:, :, 0] < 100) & (arr[:, :, 1] > 200) & (arr[:, :, 2] < 100)
    mask_orig = Image.fromarray((green * 255).astype(np.uint8), mode='L')
    bbox = mask_orig.getbbox()
    print(f"[DEBUG] bbox зелёной области: {bbox}")
    if not bbox:
        # Если зелёного не найдено, возвращаем просто шаблон (в RGB)
        print("[DEBUG] Не найдена зелёная область. Возвращаем оригинал в RGB.")
        # Сразу возвращаем содержимое в байтах
        buf = io.BytesIO()
        template_img.convert('RGB').save(buf, format='JPEG', quality=90)
        return buf.getvalue()

    # Центр прямоугольника зелёной области
    cx_orig = (bbox[0] + bbox[2]) // 2
    cy_orig = (bbox[1] + bbox[3]) // 2
    print(f"[DEBUG] Центр зелёной области (cx_orig, cy_orig): ({cx_orig}, {cy_orig})")

    # 1) Увеличиваем шаблон (апскейл) в 5 раз
    up_w, up_h = int(w * upscale_factor), int(h * upscale_factor)
    print(f"[DEBUG] Апскейл шаблона до: {up_w}x{up_h}")
    template_up = template_img.resize((up_w, up_h), Image.BICUBIC).convert('RGBA')

    # Переводим координаты зелёной области
    cx, cy = int(cx_orig * upscale_factor), int(cy_orig * upscale_factor)
    bbox_up = (
        int(bbox[0] * upscale_factor),
        int(bbox[1] * upscale_factor),
        int(bbox[2] * upscale_factor),
        int(bbox[3] * upscale_factor)
    )

    base_w = bbox_up[2] - bbox_up[0]
    base_h = bbox_up[3] - bbox_up[1]
    scale_pixels_up = int(scale_pixels * upscale_factor)
    crop_w = base_w + scale_pixels_up
    crop_h = base_h + scale_pixels_up

    # Увеличиваем фото пользователя
    user_photo_img = user_photo_img.convert('RGBA')
    p_w, p_h = user_photo_img.size
    print(f"[DEBUG] Размер входного фото пользователя: {p_w}x{p_h}")
    photo_up = user_photo_img.resize((int(p_w * upscale_factor), int(p_h * upscale_factor)), Image.BICUBIC)
    pu_w, pu_h = photo_up.size

    # Вычисляем масштаб, чтобы фото покрывало зелёную область
    sf = max(crop_w / pu_w, crop_h / pu_h)
    print(f"[DEBUG] Масштаб для фото: {sf:.2f}")
    photo_scaled = photo_up.resize((int(pu_w * sf), int(pu_h * sf)), Image.BICUBIC)
    ps_w, ps_h = photo_scaled.size

    # Делаем кроп фото под прямоугольник зелёной области (+запас)
    crop_left = (ps_w - crop_w) // 2
    crop_top = (ps_h - crop_h) // 2
    cropped_photo = photo_scaled.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))
    print(f"[DEBUG] Кроп фото до: {crop_w}x{crop_h}")

    # Случайный поворот и сдвиг
    angle = random.choice([-1, 1]) * random.randint(min_rotation, max_rotation)
    dx_ = random.randint(min_shift, max_shift) * random.choice([-1, 1])
    dy_ = random.randint(min_shift, max_shift) * random.choice([-1, 1])
    print(f"[DEBUG] Поворот: {angle}°, сдвиг: dx={dx_}, dy={dy_}")
    rotated_photo = cropped_photo.rotate(
        angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0)
    )

    # Координаты вставки в шаблон (центр зелёной области + случайный сдвиг)
    target_center = (cx + dx_, cy + dy_)
    paste_x = target_center[0] - (rotated_photo.width // 2)
    paste_y = target_center[1] - (rotated_photo.height // 2)

    # Загружаем фильтр и накладываем его поверх
    filter_img = Image.open(filter_path).convert('RGBA')
    print(f"[DEBUG] Загрузка и ресайз фильтра: {filter_path}")
    filter_resized = filter_img.resize((cropped_photo.width, cropped_photo.height + 30), Image.BICUBIC)
    rotated_filter = filter_resized.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    composite = Image.new('RGBA', rotated_filter.size, (0, 0, 0, 0))
    composite.paste(rotated_photo, (0, 0))
    final_composite = Image.alpha_composite(composite, rotated_filter)

    # Вставляем всё в увеличенный шаблон
    layer = Image.new('RGBA', template_up.size, (0, 0, 0, 0))
    layer.paste(final_composite, (paste_x, paste_y))
    final_img = Image.alpha_composite(template_up, layer)

    # Размываем левую границу вставленного фото
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

    p1, p2 = rotated_corners[0], rotated_corners[3]
    dx_edge, dy_edge = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx_edge, dy_edge)
    if length != 0:
        # Нормаль к левому краю
        nx, ny = (-dy_edge / length, dx_edge / length)
        offset = thickness / 2
        polygon = [
            (p1[0] + nx * offset, p1[1] + ny * offset),
            (p2[0] + nx * offset, p2[1] + ny * offset),
            (p2[0] - nx * offset, p2[1] - ny * offset),
            (p1[0] - nx * offset, p1[1] - ny * offset)
        ]
        mask = Image.new('L', final_img.size, 0)
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        blurred = final_img.filter(ImageFilter.BoxBlur(box_blur_radius))
        final_img = Image.composite(blurred, final_img, mask)

    # JPEG не поддерживает альфа-канал, переводим в RGB
    final_img = final_img.convert('RGB')

    # --- Дополнительный блок: Сохраняем во временный файл и закрываем, чтобы освободить память ---
    temp_filename = "temp_upscaled_output.jpg"
    print("[DEBUG] Сохраняем 5× (большое) изображение во временный файл:", temp_filename)
    final_img.save(temp_filename, format="JPEG", quality=95)
    final_img.close()  # Освобождаем память, связанную с большим изображением

    # Теперь заново открываем сохранённое изображение
    reopened_img = Image.open(temp_filename)

    # Если слишком большое, уменьшим (thumbnail) до MAX_TELEGRAM_DIM
    if reopened_img.width > MAX_TELEGRAM_DIM or reopened_img.height > MAX_TELEGRAM_DIM:
        print("[DEBUG] Слишком большое изображение, делаем thumbnail...")
        reopened_img.thumbnail((MAX_TELEGRAM_DIM, MAX_TELEGRAM_DIM))

    # Сохраняем финальный результат в буфер
    out_buf = io.BytesIO()
    reopened_img.save(out_buf, format='JPEG', quality=90)
    out_buf.seek(0)

    # Закрываем reopened_img и удаляем временный файл
    reopened_img.close()
    os.remove(temp_filename)

    print(f"[DEBUG] Завершение process_template_photo. Итоговый размер: {out_buf.getbuffer().nbytes} байт")
    # Возвращаем байты JPEG, готовые к отправке
    return out_buf.getvalue()

@bot.message_handler(commands=['start'])
def cmd_start(message):
    """
    Запуск. Показываем список доступных персонажей (например, IREN или JUL).
    """
    print("[DEBUG] /start команду вызвал пользователь:", message.chat.id)
    mapping = build_persona_actions_mapping()
    if not mapping:
        bot.send_message(message.chat.id, "Нет шаблонов.")
        return

    # Создаём инлайн-кнопки по всем персонажам
    markup = InlineKeyboardMarkup()
    for persona in sorted(mapping.keys()):
        btn = InlineKeyboardButton(persona, callback_data=f"persona_{persona}")
        markup.add(btn)

    # Отправляем пользователю сообщение с кнопками
    bot.send_message(
        message.chat.id,
        "🎭Выберите персонажа:",
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith("persona_"))
def choose_persona(call):
    """
    Выбор конкретного персонажа. После выбора → предлагаются действия.
    """
    persona = call.data.replace("persona_", "")
    print(f"[DEBUG] Пользователь {call.message.chat.id} выбрал персонажа: {persona}")
    mapping = build_persona_actions_mapping()
    if persona not in mapping:
        bot.answer_callback_query(call.id, text="Такого персонажа нет.")
        return

    user_data[call.message.chat.id] = {"persona": persona, "state": "choose_action"}
    bot.answer_callback_query(call.id, text=f"Выбран персонаж: {persona}")

    # Предлагаем действия (например, aura, finance, love...)
    actions = sorted(mapping[persona].keys())
    markup = InlineKeyboardMarkup()
    for act in actions:
        markup.add(InlineKeyboardButton(act, callback_data=f"action_{act}"))

    bot.send_message(call.message.chat.id, "🖼️Выберите шаблон:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("action_"))
def choose_action(call):
    """
    После выбора действия меняем состояние на 'waiting_photo' и ждём фото.
    """
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

@bot.message_handler(content_types=['photo', 'document'])
def handle_photo_or_document(message):
    """
    Обрабатываем как фото или документ (если пользователь присылает файл),
    пробуем открыть как изображение и вставляем в шаблон.
    """
    chat_id = message.chat.id
    print(f"[DEBUG] handle_photo_or_document от пользователя {chat_id}. Тип: {message.content_type}")

    # Проверяем, действительно ли пользователь ждёт фото
    if user_data.get(chat_id, {}).get("state") != "waiting_photo":
        print("[DEBUG] Но пользователь не в состоянии waiting_photo. Игнорируем.")
        bot.send_message(chat_id, "Сначала выберите шаблон /start.")
        return

    # Сообщаем, что идёт обработка
    bot.send_message(chat_id, "⏳Идёт обработка...")

    persona = user_data[chat_id]["persona"]
    action = user_data[chat_id]["action"]
    mapping = build_persona_actions_mapping()
    print(f"[DEBUG] У пользователя {chat_id} текущий persona={persona}, action={action}")

    # Находим соответствующий шаблон
    tmpl_filename = mapping[persona][action]
    template_path = os.path.join(templates_dir, tmpl_filename)
    if not os.path.exists(template_path):
        print("[DEBUG] Шаблон не найден:", template_path)
        bot.send_message(chat_id, "Шаблон не найден.")
        return
    print("[DEBUG] Используем шаблон:", template_path)

    # Получаем file_id для загрузки
    if message.content_type == 'photo':
        file_id = message.photo[-1].file_id
    else:
        # document
        file_id = message.document.file_id
    print(f"[DEBUG] Скачиваем файл с file_id={file_id}")

    # Скачиваем файл
    file_info = bot.get_file(file_id)
    downloaded = bot.download_file(file_info.file_path)
    user_photo_buf = io.BytesIO(downloaded)

    # Пробуем открыть как изображение
    try:
        user_photo = Image.open(user_photo_buf)
        print(f"[DEBUG] Успешно открыли фото. Размер: {user_photo.size}")
    except Exception as e:
        print("[DEBUG] Не удалось открыть файл как изображение:", e)
        bot.send_message(chat_id, "Не удалось открыть файл как изображение.")
        return

    # Открываем шаблон и обрабатываем
    template_img = Image.open(template_path)
    print(f"[DEBUG] Открыли шаблон. Размер: {template_img.size}")

    # Создаём итоговое изображение (в виде байтов)
    try:
        result_bytes = process_template_photo(template_img, user_photo)
    except Exception as e:
        print("[DEBUG] Ошибка при процессинге фото:", e)
        traceback.print_exc()
        bot.send_message(chat_id, "Произошла ошибка при обработке изображения.")
        return

    # Отправляем результат
    # Кнопка "Создать ещё" (возвращает к /start)
    again_markup = InlineKeyboardMarkup()
    again_markup.add(InlineKeyboardButton("🔄Сгенерировать фото", callback_data="create_more"))

    print(f"[DEBUG] Отправляем конечное изображение пользователю {chat_id}.")
    bot.send_photo(chat_id, result_bytes, caption="✅ Готово!", reply_markup=again_markup)

    # Сбрасываем состояние
    user_data[chat_id]["state"] = None

    # Закрываем открытые объекты
    user_photo.close()
    template_img.close()

@bot.callback_query_handler(func=lambda call: call.data == "create_more")
def callback_create_more(call):
    """
    При нажатии на "Создать ещё" просто повторяем процесс (вызываем /start).
    """
    print("[DEBUG] callback_create_more: пользователь хочет создать ещё.")
    bot.answer_callback_query(call.id)
    cmd_start(call.message)

if __name__ == "__main__":
    print("[DEBUG] Удаляем вебхук, если был установлен.")
    bot.remove_webhook()
    print("[DEBUG] Запуск bot.infinity_polling()")

    # Обёртка, чтобы бот перезапускался при неперехваченных ошибках
    while True:
        try:
            bot.infinity_polling()
        except Exception as e:
            print("[FATAL ERROR]", e)
            traceback.print_exc()
            print("Перезапуск через 5 секунд...")
            time.sleep(5)

