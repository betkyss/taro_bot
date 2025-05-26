import os
import io
import traceback
import math
import random

import numpy as np
import cv2  # OpenCV для поиска повёрнутого прямоугольника
from PIL import Image, ImageDraw, ImageFilter
from dotenv import load_dotenv
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, Message

############################################
# 1) ЗАГРУЗКА БЕЛОГО СПИСКА ПОЛЬЗОВАТЕЛЕЙ
############################################
def load_allowed_user_ids(filename="allowed_users.txt"):
    """
    Читает файл allowed_users.txt и возвращает множество user_id (int).
    Игнорирует пустые строки и строки, начинающиеся с '#'.
    """
    allowed = set()
    if not os.path.exists(filename):
        print(f"[DEBUG] {filename} не найден, белый список пуст.")
        return allowed
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                allowed.add(int(line))
            except ValueError:
                print(f"[DEBUG] Неправильный ID в {filename}: {line}")
    print(f"[DEBUG] Загружено {len(allowed)} ID из {filename}")
    return allowed

ALLOWED_USER_IDS = load_allowed_user_ids()

############################################
# 2) ЗАГРУЗКА ТОКЕНА И ИНИЦИАЛИЗАЦИЯ БОТА
############################################
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

# увеличить таймауты на отправку файлов
import telebot.apihelper
telebot.apihelper.SEND_FILE_TIMEOUT = 120
telebot.apihelper.CONNECT_TIMEOUT = 30
telebot.apihelper.READ_TIMEOUT = 30

print("Бот запущен и готов к работе")

############################################
# 3) КОНСТАНТЫ И НАСТРОЙКИ
############################################
templates_dir   = "templates"     # папка с шаблонами
filter_path     = "filter.png"    # фильтр, который накладывается поверх фото
MAX_TELEGRAM_DIM = 4096           # максимальная длина стороны для Телеграм

# параметры случайного смещения и поворота
min_shift, max_shift       = 5, 15
min_rotation, max_rotation = 2, 4
# сколько пикселей расширять прямоугольник
scale_pixels = 35
# множитель для внутреннего «апскейла» (чтобы не терять качество при трансформациях)
upscale_factor = 5.0
# параметры для размывания края
thickness       = 25
box_blur_radius = 5

# хранение состояния для каждого чата
user_data = {}

############################################
# 4) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
############################################
def get_personas():
    """
    Возвращает список папок (персонажей) внутри templates/
    """
    return sorted([
        d for d in os.listdir(templates_dir)
        if os.path.isdir(os.path.join(templates_dir, d))
    ])

def build_persona_actions_mapping():
    """
    Для каждого персонажа (кроме JUL) собирает доступные шаблоны действий.
    Ожидает файлы вида PERSONA_action.ext.
    Возвращает dict { persona: { action: filename, ... } }.
    """
    mapping = {}
    for persona in get_personas():
        if persona == "JUL":
            continue
        persona_folder = os.path.join(templates_dir, persona)
        actions = {}
        for fname in os.listdir(persona_folder):
            name, ext = os.path.splitext(fname)
            parts = name.split("_", 1)
            if len(parts) != 2:
                continue
            _, action = parts
            if ext.lower() in (".jpg", ".jpeg", ".png"):
                actions[action] = fname
        if actions:
            mapping[persona] = actions
    return mapping

############################################
# 5) ФУНКЦИЯ ОБРАБОТКИ ИЗОБРАЖЕНИЯ
############################################
def process_template_photo(template_img: Image.Image,
                           user_photo_img: Image.Image) -> bytes:
    """
    1. Находит зелёный прямоугольник (по маске).
    2. Определяет его размеры и угол относительно вертикали (0° — фото «как есть»).
    3. Создаёт монолит (фото пользователя + filter.png) портретной ориентации,
       размером (long+scale_pixels) × (short+scale_pixels) без внутренних поворотов.
    4. Поворачивает монолит на найденный угол и кладёт по центру зелёной области.
    """
    # ── константы
    upscale_factor   = 5.0
    scale_pixels     = 35
    MAX_TELEGRAM_DIM = 4096
    thickness        = 25
    box_blur_radius  = 5
    filter_path      = "filter.png"

    # ── 1. берём шаблон как RGBA, создаём только маску зелёных пикселей
    tpl_rgba = template_img.convert("RGBA")
    tpl_arr  = np.asarray(tpl_rgba).copy()        # copy() -> массив станет записываемым
    b, g, r, _ = tpl_arr.transpose(2, 0, 1)

    green_mask = ((g > 200) & (r < 100) & (b < 100)).astype(np.uint8) * 255  # 0/255

    # ── 2. находим минимальный повёрнутый прямоугольник по маске
    contours, _ = cv2.findContours(green_mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:                              # если маски нет — вернём оригинал
        buf = io.BytesIO()
        template_img.convert("RGB").save(buf, "JPEG", quality=90)
        buf.seek(0)
        return buf.getvalue()

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))   # (cx,cy), (w,h), angle
    box  = cv2.boxPoints(rect)                                   # 4 точки float32

    # длинное ребро → его длина и угол
    edges    = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    lengths  = [np.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]) for e in edges]
    (x1, y1), (x2, y2) = edges[int(np.argmax(lengths))]
    long_dim, short_dim = max(lengths), min(lengths)
    cx0, cy0 = rect[0]

    angle_long = math.degrees(math.atan2(y2 - y1, x2 - x1))  # угол длинной стороны (горизонт)

    # ── 2a. угол, на который надо повернуть монолит, чтобы вертикаль фото = 0°
    rotation_angle = angle_long - 90
    if rotation_angle > 90:
        rotation_angle -= 180
    elif rotation_angle < -90:
        rotation_angle += 180
    # теперь rotation_angle ∈ (-90 ; 90]
    print(rotation_angle)

    # ── 3. апскейл шаблона
    up_w, up_h = (np.array(template_img.size) * upscale_factor).astype(int)
    tpl_up = tpl_rgba.resize((up_w, up_h), Image.BICUBIC)
    cx, cy = cx0 * upscale_factor, cy0 * upscale_factor

    # ── 4. размеры области под фото + фильтр
    area_h = int((long_dim  + scale_pixels) * upscale_factor)   # высота
    area_w = int((short_dim + scale_pixels) * upscale_factor)   # ширина

    # ── 5. фото пользователя (без поворота)
    user_rgba = user_photo_img.convert("RGBA")
    user_up   = user_rgba.resize(
        (int(user_rgba.width * upscale_factor),
         int(user_rgba.height * upscale_factor)),
        Image.BICUBIC
    )
    scale = max(area_w / user_up.width, area_h / user_up.height)
    user_fit = user_up.resize(
        (int(user_up.width * scale), int(user_up.height * scale)),
        Image.BICUBIC
    )
    left = (user_fit.width  - area_w) // 2
    top  = (user_fit.height - area_h) // 2
    cropped = user_fit.crop((left, top, left + area_w, top + area_h))

    # ── 6. фильтр
    filt  = Image.open(filter_path).convert("RGBA")
    filt2 = filt.resize((area_w, area_h + 30), Image.BICUBIC)

    monolith = Image.new("RGBA", filt2.size, (0, 0, 0, 0))
    monolith.paste(cropped, (0, 0), cropped)
    monolith = Image.alpha_composite(monolith, filt2)

    # ── 7. поворачиваем монолит
    monolith_rot = monolith.rotate(
        -rotation_angle,
        expand=True,
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0, 0)
    )

    # ── 8. вставка по центру зелёной области
    layer = Image.new("RGBA", tpl_up.size, (0, 0, 0, 0))
    paste_x = int(cx - monolith_rot.width  / 2)
    paste_y = int(cy - monolith_rot.height / 2)
    layer.paste(monolith_rot, (paste_x, paste_y), monolith_rot)
    result = Image.alpha_composite(tpl_up, layer)

    # ── 9. размытие левой кромки
    if thickness > 0 and box_blur_radius > 0:
        strip = result.crop((0, 0, thickness, result.height))
        result.paste(strip.filter(ImageFilter.BoxBlur(box_blur_radius)), (0, 0))

    # ── 10. ограничение Telegram
    if max(result.size) > MAX_TELEGRAM_DIM:
        result.thumbnail((MAX_TELEGRAM_DIM, MAX_TELEGRAM_DIM), Image.LANCZOS)

    # ── 11. JPEG-байты
    buf = io.BytesIO()
    result.convert("RGB").save(buf, "JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue()

############################################
# 6) ХЕНДЛЕРЫ ТЕЛЕБОТА (БЕЗ ИЗМЕНЕНИЙ)
############################################
@bot.message_handler(commands=["start"])
def cmd_start(message: Message):
    chat_id = message.chat.id
    if message.from_user.id not in ALLOWED_USER_IDS:
        bot.send_message(chat_id, "У вас нет доступа к боту.")
        return

    personas = get_personas()
    if not personas:
        bot.send_message(chat_id, "Нет доступных шаблонов.")
        return

    markup = InlineKeyboardMarkup()
    for p in personas:
        markup.add(InlineKeyboardButton(p, callback_data=f"persona_{p}"))
    bot.send_message(chat_id, "Выберите персонажа:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("persona_"))
def handle_persona_callback(call):
    chat_id = call.message.chat.id
    if call.from_user.id not in ALLOWED_USER_IDS:
        bot.answer_callback_query(call.id, "Нет доступа.")
        return

    persona = call.data.split("_",1)[1]
    user_data[chat_id] = {"persona": persona}

    if persona == "JUL":
        user_data[chat_id]["state"] = "choose_stage"
        markup = InlineKeyboardMarkup()
        options = [
            ("0", "0 – бесплатный расклад"),
            ("1", "1 – результат платной диагностики"),
            ("2", "2 – результат после ритуала"),
            ("3", "3 – результат после ритуала"),
        ]
        for val, lbl in options:
            markup.add(InlineKeyboardButton(lbl, callback_data=f"stage_{val}"))
        bot.send_message(chat_id, "Выберите этап:", reply_markup=markup)
    else:
        mapping = build_persona_actions_mapping()
        actions = sorted(mapping.get(persona, {}).keys())
        user_data[chat_id]["state"] = "choose_action"
        markup = InlineKeyboardMarkup()
        for act in actions:
            markup.add(InlineKeyboardButton(act, callback_data=f"action_{act}"))
        bot.send_message(chat_id, "Выберите шаблон:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("stage_"))
def handle_stage_callback(call):
    chat_id = call.message.chat.id
    stage = call.data.split("_",1)[1]
    user_data[chat_id]["stage"] = stage

    if stage in ("1", "2"):
        user_data[chat_id]["state"] = "waiting_photo"
        bot.send_message(chat_id, "Отправьте фото для вставки")
    else:
        folder = os.path.join(templates_dir, "JUL", stage)
        try:
            files = [f for f in os.listdir(folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        except FileNotFoundError:
            files = []
        actions = sorted({os.path.splitext(f)[0].split("_",1)[1] for f in files})
        user_data[chat_id]["state"] = "choose_action"
        markup = InlineKeyboardMarkup()
        for act in actions:
            markup.add(InlineKeyboardButton(act, callback_data=f"action_{act}"))
        bot.send_message(chat_id, "Выберите шаблон:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("action_"))
def handle_action_callback(call):
    chat_id = call.message.chat.id
    action = call.data.split("_",1)[1]
    user_data[chat_id]["action"] = action
    user_data[chat_id]["state"] = "waiting_photo"
    bot.send_message(chat_id, "Отправьте фото для вставки")

@bot.callback_query_handler(func=lambda call: call.data == "create_more")
def handle_create_more(call):
    cmd_start(call.message)

@bot.message_handler(content_types=["photo", "document"])
def handle_media(message: Message):
    chat_id = message.chat.id
    if user_data.get(chat_id, {}).get("state") != "waiting_photo":
        bot.send_message(chat_id, "Сначала выберите шаблон через /start.")
        return

    bot.send_message(chat_id, "Обработка...")
    info = user_data[chat_id]
    persona = info["persona"]
    stage = info.get("stage")
    action = info.get("action")

    # определяем путь к шаблону
    if persona == "JUL":
        if stage in ("1", "2"):
            template_path = os.path.join(templates_dir, "JUL", stage, "JUL.jpeg")
        else:
            base = os.path.join(templates_dir, "JUL", stage, f"JUL_{action}")
            template_path = next(
                (base + ext for ext in (".jpg", ".jpeg", ".png") if os.path.exists(base + ext)),
                None
            )
    else:
        mapping = build_persona_actions_mapping()
        fname = mapping.get(persona, {}).get(action)
        template_path = os.path.join(templates_dir, persona, fname) if fname else None

    if not template_path or not os.path.exists(template_path):
        bot.send_message(chat_id, "Шаблон не найден.")
        return

    # загрузка фото пользователя
    if message.content_type == "photo":
        file_id = message.photo[-1].file_id
    else:
        file_id = message.document.file_id
    file_info = bot.get_file(file_id)
    downloaded = bot.download_file(file_info.file_path)
    try:
        user_img = Image.open(io.BytesIO(downloaded))
    except Exception:
        bot.send_message(chat_id, "Невозможно открыть изображение.")
        return

    # вставка фото
    try:
        template_img = Image.open(template_path)
        result_bytes = process_template_photo(template_img, user_img)
    except Exception:
        print("[ERROR]", traceback.format_exc())
        bot.send_message(chat_id, "Ошибка при обработке.")
        return

    # отправка результата
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Сгенерировать снова", callback_data="create_more"))
    bot.send_photo(chat_id, result_bytes, caption="Готово!", reply_markup=markup)
    user_data[chat_id]["state"] = None

############################################
# 7) ЗАПУСК БОТА
############################################
if __name__ == "__main__":
    bot.remove_webhook()
    bot.infinity_polling()

