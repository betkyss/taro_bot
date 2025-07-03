#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram-бот для вставки фотографии пользователя в шаблон (зелёный/фильтр).
"""

# ────────────────────────────────────────────────────────────────────
# ► 0. ИМПОРТЫ
# ────────────────────────────────────────────────────────────────────
import io, os, math, random, traceback, warnings, sys, logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageFilter
from dotenv import load_dotenv

os.makedirs("logs", exist_ok=True)

# настраиваем корневой логгер
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)


import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, Message

# ────────────────────────────────────────────────────────────────────
# ► 1. НАСТРОЙКА PIL  (отключаем «бомбу» на большие изображения)
# ────────────────────────────────────────────────────────────────────
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# ────────────────────────────────────────────────────────────────────
# ► 2. ГЛОБАЛЬНЫЕ КОНСТАНТЫ
# ────────────────────────────────────────────────────────────────────
templates_dir      = "templates"               # папка с шаблонами
filter_path        = "filter.png"            # PNG-фильтр (с тенью 22 px)
OUT_DIM            = 2048                      # итоговая длинная сторона
scale_pixels       = 75                       # «запас» к прямоугольнику
upscale_factor     = 0.5                       # внутренний апскейл
SCALE_MONO         = 8                         # апскейл монолита
min_shift, max_shift       = 2, 8             # случайный сдвиг
min_rotation, max_rotation = 1, 3              # случайный угол
thickness, box_blur_radius = 25, 5             # бокс-блюр левого края
MAX_PIXELS_TPL     = 80_000_000                # потолок апскейла шаблона
TG_PHOTO_LIMIT     = 10_485_760                # 10 МБ (Telegram)
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ────────────────────────────────────────────────────────────────────
# ► 3. ГОТОВЫЕ СООБЩЕНИЯ / ЭМОДЗИ
# ────────────────────────────────────────────────────────────────────
MSG_SELECT_PERSONA   = "🎭 Выберите персонажа:"
MSG_SELECT_STAGE     = "📟 Выберите этап"
MSG_SELECT_TEMPLATE  = "🖼️ Выберите шаблон"
MSG_SEND_PHOTO       = "📥 Отправь фото для вставки"
MSG_PROCESSING       = "⏳ Идёт обработка…"
MSG_DONE             = "✅ Готово!"
BTN_REGENERATE       = "Сгенерировать снова"
MSG_NO_ACCESS        = "Нет доступа."
MSG_START_FIRST      = "Сначала /start."
MSG_TEMPLATE_NOT_FOUND = "Шаблон не найден."
MSG_ERROR_INTERNAL   = "⚠️ Внутренняя ошибка. Посмотрите логи."
MSG_NO_TEMPLATES_FOUND = "Не найдено подходящих шаблонов для этого выбора."

# Этот словарь теперь используется для получения красивых имен этапов
STAGE_NAME_MAP = {
    "0": "0️⃣ (бесплатный расклад)",
    "1": "1️⃣ (рез-т платной д-ки)",
    "2": "2️⃣ (рез-т после ритуала)",
    "3": "3️⃣ (рез-т после ритуала)",
}

# ────────────────────────────────────────────────────────────────────
# ► 4. ЗАГРУЗКА «БЕЛОГО» СПИСКА ПОЛЬЗОВАТЕЛЕЙ (who can use the bot)
# ────────────────────────────────────────────────────────────────────
def load_allowed_user_ids(fname: str = "allowed_users.txt") -> set[int]:
    """Считывает allowed_users.txt → множество user_id (int)."""
    ids = set()
    if os.path.exists(fname):
        with open(fname, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        ids.add(int(line))
                    except ValueError:
                        logging.warning(f"Не удалось прочитать User ID: {line} из {fname}")
    logging.info(f"Белый список: {len(ids)} ID загружено.")
    return ids


ALLOWED_USER_IDS = load_allowed_user_ids()

# ────────────────────────────────────────────────────────────────────
# ► 5. ИНИЦИАЛИЗАЦИЯ TELEGRAM-БОТА
# ────────────────────────────────────────────────────────────────────
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    logging.critical("BOT_TOKEN не найден в .env файле! Бот не сможет запуститься.")
    sys.exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

# увеличиваем таймауты отправки файлов
import telebot.apihelper as tbh

tbh.SEND_FILE_TIMEOUT = 120
tbh.CONNECT_TIMEOUT = 30
tbh.READ_TIMEOUT = 30

logging.info("Bot ready")

# ────────────────────────────────────────────────────────────────────
# ► 6. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (динамическое определение структуры)
# ────────────────────────────────────────────────────────────────────
def get_personas() -> list[str]:
    """Получить список подпапок внутри templates/ (персонажей)."""
    if not os.path.exists(templates_dir) or not os.path.isdir(templates_dir):
        logging.error(f"Папка шаблонов '{templates_dir}' не найдена.")
        return []
    return sorted(
        d for d in os.listdir(templates_dir)
        if os.path.isdir(os.path.join(templates_dir, d))
    )

def get_stages_for_persona(persona: str) -> List[Tuple[str, str]]:
    """
    Получает список этапов для персонажа.
    Этапы - это подпапки с числовыми именами.
    Возвращает список кортежей (имя_папки, отображаемое_имя).
    """
    persona_path = os.path.join(templates_dir, persona)
    stages = []
    if os.path.isdir(persona_path):
        for item in os.listdir(persona_path):
            item_path = os.path.join(persona_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                display_name = STAGE_NAME_MAP.get(item, f"Этап {item}")
                stages.append((item, display_name))
    return sorted(stages, key=lambda x: int(x[0])) # Сортируем по числовому значению

def get_templates_in_path(current_path: str) -> List[str]:
    """
    Получает список файлов шаблонов (картинок) в указанной папке.
    Возвращает список имен файлов (без пути, но с расширением).
    """
    templates = []
    if os.path.isdir(current_path):
        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            if os.path.isfile(item_path) and item.lower().endswith(VALID_IMAGE_EXTENSIONS):
                templates.append(item)
    return sorted(templates)

def get_template_name_without_extension(filename: str) -> str:
    """Удаляет расширение из имени файла."""
    return os.path.splitext(filename)[0]

# --- Функции для обработки изображений (оставлены как есть, но можно вынести в отдельный модуль) ---
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Сортировка 4 точек → TL, TR, BR, BL (для getPerspective)."""
    pts = np.asarray(pts, dtype="float32")
    s = pts.sum(1)
    diff = np.diff(pts, axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def _save_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf.getvalue()

def _save_jpeg(img: Image.Image, q: int = 95) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    buf.seek(0)
    return buf.getvalue()

# ────────────────────────────────────────────────────────────────────
# ► 7. ОСНОВНАЯ ОБРАБОТКА ИЗОБРАЖЕНИЯ
# ────────────────────────────────────────────────────────────────────
def process_template_photo(tpl_img: Image.Image, user_img: Image.Image) -> bytes:
    """
    Вставляет user_img в зелёный прямоугольник tpl_img и возвращает bytes.
    Логика подробно прокомментирована внутри.
    """
    # 7.1 --- АПСКЕЙЛ ШАБЛОНА ДО OUT_DIM (но не более 80 Мп)
    out_scale = OUT_DIM / max(tpl_img.size) if max(tpl_img.size) < OUT_DIM else 1.0
    tpl_big = tpl_img.resize(
        (int(tpl_img.width * out_scale), int(tpl_img.height * out_scale)),
        Image.LANCZOS
    )
    # ограничиваем до 80 Мп (RAM-safety)
    if tpl_big.width * tpl_big.height > MAX_PIXELS_TPL:
        factor = math.sqrt(MAX_PIXELS_TPL / (tpl_big.width * tpl_big.height))
        tpl_big = tpl_big.resize(
            (int(tpl_big.width * factor), int(tpl_big.height * factor)),
            Image.LANCZOS
        )
        out_scale *= factor

    tpl_rgba = tpl_big.convert("RGBA")

    # 7.2 --- ПОИСК ЗЕЛЁНОЙ ОБЛАСТИ
    b, g, r, _ = np.asarray(tpl_rgba).transpose(2, 0, 1)
    mask = ((g > 200) & (r < 100) & (b < 100)).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _save_png(tpl_rgba)  # зелёного нет → отдаём шаблон без изменений
    cnt = max(cnts, key=cv2.contourArea)

    # 7.3 --- ПРОВЕРКА: 4-угольник или нет
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    persp = len(approx) == 4  # True → перспективная вставка

    if persp:
        # • координаты 4 точек + «запас»
        quad = order_corners([p[0] for p in approx])
        center = quad.mean(0, keepdims=True)
        vecs = quad - center
        lens = np.linalg.norm(vecs, 1, keepdims=True)
        quad = quad + vecs / (lens + 1e-6) * scale_pixels * out_scale

        # размеры прямоугольника
        wA, hA = np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[0] - quad[3])
        wB, hB = np.linalg.norm(quad[2] - quad[3]), np.linalg.norm(quad[1] - quad[2])
        long_side, short_side = int(max(hA, hB)), int(max(wA, wB))
    else:
        # fallback: minAreaRect
        (cx0, cy0), (w0, h0), ang = cv2.minAreaRect(cnt)
        long_side, short_side = int(max(w0, h0)), int(min(w0, h0))

    # 7.4 --- СОЗДАЁМ «МОНОЛИТ» (фото + фильтр) С НУЖНЫМИ РАЗМЕРАМИ
    H = int((long_side + scale_pixels * out_scale) * upscale_factor) * SCALE_MONO
    W = int((short_side + scale_pixels * out_scale) * upscale_factor) * SCALE_MONO

    # ► 7.4.1 фото пользователя – масштаб «по длинной стороне» + центр-обрезка
    usr_big = user_img.convert("RGBA").resize(
        (int(user_img.width * upscale_factor * SCALE_MONO),
         int(user_img.height * upscale_factor * SCALE_MONO)),
        Image.LANCZOS
    )
    # берём большее из коэффициентов → заполняем всю область
    sc_fill = max(W / usr_big.width, H / usr_big.height)
    usr_fill = usr_big.resize(
        (int(usr_big.width * sc_fill), int(usr_big.height * sc_fill)),
        Image.BICUBIC if sc_fill > 1 else Image.LANCZOS
    )
    lft = (usr_fill.width - W) // 2
    top = (usr_fill.height - H) // 2
    cropped = usr_fill.crop((lft, top, lft + W, top + H))

    # ► 7.4.2 PNG-фильтр с прозрачной тенью 22 px
    if not os.path.exists(filter_path):
        logging.error(f"Файл фильтра {filter_path} не найден!")
        # Можно либо вернуть ошибку, либо продолжить без фильтра
        # Для примера, продолжим без фильтра, но залогируем
        mono = cropped # Без фильтра
    else:
        filt = Image.open(filter_path).convert("RGBA").resize((W, H), Image.LANCZOS)
        # ► 7.4.3 композим фото + фильтр
        mono = Image.new("RGBA", filt.size, (0, 0, 0, 0))
        mono.paste(cropped, (0, 0)) # Убрали cropped из paste, т.к. он уже RGBA
        mono = Image.alpha_composite(mono, filt)


    # ► 7.4.4 небольшой случайный поворот «монолита»
    mono = mono.rotate(
        random.choice([-1, 1]) * random.uniform(min_rotation, max_rotation),
        expand=True,
        resample=Image.BICUBIC
    )

    # после вращения уменьшаем до «обычного» масштаба
    mono = mono.resize((mono.width // SCALE_MONO, mono.height // SCALE_MONO), Image.LANCZOS)

    # случайный сдвиг +/- dx,dy
    dx = random.choice([-1, 1]) * random.randint(min_shift, max_shift)
    dy = random.choice([-1, 1]) * random.randint(min_shift, max_shift)

    # 7.5 --- ВСТАВКА «МОНОЛИТА» В ШАБЛОН
    if persp:
        # ► Перспективная вставка (4 точки)
        src = np.array(
            [
                [0, 0],
                [W // SCALE_MONO, 0],
                [W // SCALE_MONO, (H + 30) // SCALE_MONO], # +30 было, оставляем для совместимости
                [0, (H + 30) // SCALE_MONO],
            ],
            dtype="float32",
        )
        quad_shift = quad + np.array([dx, dy], dtype="float32")
        M = cv2.getPerspectiveTransform(src, quad_shift)

        canvas_bgr = cv2.cvtColor(np.asarray(tpl_rgba), cv2.COLOR_RGBA2BGRA)
        mono_bgr = cv2.cvtColor(np.asarray(mono), cv2.COLOR_RGBA2BGRA)

        warp = cv2.warpPerspective(
            mono_bgr,
            M,
            dsize=tpl_rgba.size, # dsize должен быть (width, height)
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        alpha = warp[:, :, 3:4] / 255.0
        canvas_bgr[:, :, :3] = canvas_bgr[:, :, :3] * (1 - alpha) + warp[:, :, :3] * alpha
        res = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGRA2RGBA), "RGBA")
    else:
        # ► minAreaRect (почти как в старом коде, но с актуальным dx/dy)
        (cx, cy), (w_rect, h_rect), ang_rect = cv2.minAreaRect(cnt) # переименовал w0, h0, ang
        
        effective_angle = ang_rect
        if w_rect < h_rect: 
             mono_rot = mono.rotate(-effective_angle, expand=True, resample=Image.BICUBIC)
        else: 
             mono_rot = mono.rotate(-effective_angle - 90, expand=True, resample=Image.BICUBIC)

        layer = Image.new("RGBA", tpl_rgba.size, (0, 0, 0, 0))
        paste_x = int(cx + dx - mono_rot.width / 2)
        paste_y = int(cy + dy - mono_rot.height / 2)
        layer.paste(mono_rot, (paste_x, paste_y), mono_rot)
        res = Image.alpha_composite(tpl_rgba, layer)

    # 7.6 --- РАЗМЫТИЕ ЛЕВОЙ КРОМКИ
    if thickness > 0 and box_blur_radius > 0: 
        if res.width >= thickness : 
            strip = res.crop((0, 0, thickness, res.height))
            res.paste(strip.filter(ImageFilter.BoxBlur(box_blur_radius)), (0, 0))
        else:
            logging.warning("Толщина для размытия больше ширины изображения, размытие пропущено.")


    # 7.7 --- ПАКОВКА (PNG ≤10 МБ, иначе JPEG с подбором качества)
    png_bytes = _save_png(res)
    if len(png_bytes) <= TG_PHOTO_LIMIT:
        return png_bytes

    for q in (95, 90, 85, 80, 75, 70, 65):
        jpg_bytes = _save_jpeg(res, q)
        if len(jpg_bytes) <= TG_PHOTO_LIMIT:
            return jpg_bytes
    return _save_jpeg(res, 50)  # fallback


# ────────────────────────────────────────────────────────────────────
# ► 8. ХРАНИЛИЩЕ СОСТОЯНИЙ ПОЛЬЗОВАТЕЛЕЙ (chat_id → dict)
# ────────────────────────────────────────────────────────────────────
user_state: Dict[int, Dict[str, Any]] = {}

# ────────────────────────────────────────────────────────────────────
# ► 9. ХЕНДЛЕРЫ БОТА
# ────────────────────────────────────────────────────────────────────
def common_access_check(user_id: int, chat_id: int) -> bool:
    """Общая проверка доступа для хендлеров."""
    if user_id not in ALLOWED_USER_IDS:
        bot.send_message(chat_id, MSG_NO_ACCESS)
        return False
    return True

def common_access_check_callback(call) -> bool:
    """Общая проверка доступа для callback хендлеров."""
    if call.from_user.id not in ALLOWED_USER_IDS:
        bot.answer_callback_query(call.id, MSG_NO_ACCESS, show_alert=True)
        return False
    return True

def get_display_template_name(tpl_file_name: str, persona_name_str: str) -> str:
    """Формирует отображаемое имя шаблона, удаляя префикс персонажа."""
    base_name = get_template_name_without_extension(tpl_file_name)
    prefix_to_remove = f"{persona_name_str}_"
    display_name = base_name
    if base_name.startswith(prefix_to_remove):
        display_name = base_name[len(prefix_to_remove):]
    return display_name.capitalize()


@bot.callback_query_handler(func=lambda c: c.data == "start_over") 
def cb_start_again(call) -> None:
    """Перезапуск по кнопке — аналог /start."""
    if not common_access_check_callback(call):
        return
    bot.answer_callback_query(call.id)
    cmd_start(call.message, from_callback=True)


@bot.message_handler(commands=["start"])
def cmd_start(msg: Message, from_callback: bool = False) -> None:
    """Старт: выбор персонажа."""
    if not from_callback and not common_access_check(msg.from_user.id, msg.chat.id):
        return

    chat_id = msg.chat.id
    user_state[chat_id] = {} 

    personas = get_personas()
    if not personas:
        bot.send_message(chat_id, "Персонажи не найдены. Проверьте конфигурацию.")
        return

    kb = InlineKeyboardMarkup()
    for p_name in personas:
        kb.add(InlineKeyboardButton(p_name, callback_data=f"persona_{p_name}"))
    
    # Если сообщение пришло от callback (т.е. это перезапуск), редактируем его
    if from_callback:
        try:
            bot.edit_message_text(MSG_SELECT_PERSONA, chat_id, msg.message_id, reply_markup=kb)
        except Exception as e: # Если не получилось отредактировать (например, сообщение слишком старое), отправляем новое
            logging.warning(f"Не удалось отредактировать сообщение при перезапуске: {e}")
            bot.send_message(chat_id, MSG_SELECT_PERSONA, reply_markup=kb)
    else: # Иначе, отправляем новое сообщение
        bot.send_message(chat_id, MSG_SELECT_PERSONA, reply_markup=kb)


@bot.callback_query_handler(func=lambda c: c.data.startswith("persona_"))
def cb_persona(call) -> None:
    """После выбора персонажа."""
    if not common_access_check_callback(call):
        return

    chat_id = call.message.chat.id
    persona_name = call.data.split("_", 1)[1]
    user_state[chat_id] = {"persona": persona_name} 

    bot.answer_callback_query(call.id) 

    persona_path = os.path.join(templates_dir, persona_name)
    stages = get_stages_for_persona(persona_name)

    if stages:
        kb = InlineKeyboardMarkup()
        for stage_val, stage_label in stages:
            kb.add(InlineKeyboardButton(stage_label, callback_data=f"stage_{stage_val}"))
        user_state[chat_id]["state"] = "choosing_stage"
        bot.edit_message_text(MSG_SELECT_STAGE, chat_id, call.message.message_id, reply_markup=kb)
    else:
        templates_list = get_templates_in_path(persona_path)
        if len(templates_list) == 1:
            user_state[chat_id]["template_file"] = os.path.join(persona_name, templates_list[0])
            user_state[chat_id]["state"] = "waiting_photo"
            bot.edit_message_text(MSG_SEND_PHOTO, chat_id, call.message.message_id)
        elif len(templates_list) > 1:
            kb = InlineKeyboardMarkup()
            for tpl_file in templates_list:
                ### ИЗМЕНЕНИЕ ###
                # Используем новую функцию для получения имени кнопки
                button_label = get_display_template_name(tpl_file, persona_name)
                kb.add(InlineKeyboardButton(button_label, callback_data=f"template_{persona_name}/{tpl_file}"))
            user_state[chat_id]["state"] = "choosing_template"
            bot.edit_message_text(MSG_SELECT_TEMPLATE, chat_id, call.message.message_id, reply_markup=kb)
        else:
            bot.edit_message_text(MSG_NO_TEMPLATES_FOUND, chat_id, call.message.message_id)
            logging.warning(f"Нет шаблонов в {persona_path} для персонажа {persona_name}")


@bot.callback_query_handler(func=lambda c: c.data.startswith("stage_"))
def cb_stage(call) -> None:
    """После выбора этапа."""
    if not common_access_check_callback(call):
        return

    chat_id = call.message.chat.id
    stage_val = call.data.split("_", 1)[1]

    if "persona" not in user_state.get(chat_id, {}):
        bot.answer_callback_query(call.id, "Ошибка состояния, начните заново /start", show_alert=True)
        cmd_start(call.message, from_callback=True) 
        return

    user_state[chat_id]["stage"] = stage_val
    bot.answer_callback_query(call.id)

    persona_name = user_state[chat_id]["persona"]
    current_path = os.path.join(templates_dir, persona_name, stage_val)
    templates_list = get_templates_in_path(current_path)

    if len(templates_list) == 1:
        user_state[chat_id]["template_file"] = os.path.join(persona_name, stage_val, templates_list[0])
        user_state[chat_id]["state"] = "waiting_photo"
        bot.edit_message_text(MSG_SEND_PHOTO, chat_id, call.message.message_id)
    elif len(templates_list) > 1:
        kb = InlineKeyboardMarkup()
        for tpl_file in templates_list:
            ### ИЗМЕНЕНИЕ ###
            # Используем новую функцию для получения имени кнопки
            button_label = get_display_template_name(tpl_file, persona_name)
            kb.add(InlineKeyboardButton(button_label, callback_data=f"template_{persona_name}/{stage_val}/{tpl_file}"))
        user_state[chat_id]["state"] = "choosing_template"
        bot.edit_message_text(MSG_SELECT_TEMPLATE, chat_id, call.message.message_id, reply_markup=kb)
    else:
        bot.edit_message_text(MSG_NO_TEMPLATES_FOUND, chat_id, call.message.message_id)
        logging.warning(f"Нет шаблонов в {current_path} для персонажа {persona_name}, этап {stage_val}")


@bot.callback_query_handler(func=lambda c: c.data.startswith("template_"))
def cb_template_selection(call) -> None:
    """После выбора конкретного шаблона (если их было несколько)."""
    if not common_access_check_callback(call):
        return

    chat_id = call.message.chat.id
    template_relative_path = call.data.split("_", 1)[1]
    
    if "persona" not in user_state.get(chat_id, {}):
        bot.answer_callback_query(call.id, "Ошибка состояния, начните заново /start", show_alert=True)
        cmd_start(call.message, from_callback=True) 
        return

    user_state[chat_id]["template_file"] = template_relative_path
    user_state[chat_id]["state"] = "waiting_photo"
    bot.answer_callback_query(call.id)
    bot.edit_message_text(MSG_SEND_PHOTO, chat_id, call.message.message_id)


@bot.message_handler(content_types=["photo", "document"])
def handle_media(msg: Message) -> None:
    """Обрабатывает входящее фото (или документ-картинку)."""
    chat_id = msg.chat.id
    if not common_access_check(msg.from_user.id, chat_id):
        return

    current_user_state = user_state.get(chat_id, {})
    if current_user_state.get("state") != "waiting_photo":
        bot.send_message(chat_id, MSG_START_FIRST)
        return

    if "template_file" not in current_user_state:
        logging.error(f"Состояние waiting_photo, но template_file не установлен для чата {chat_id}")
        bot.send_message(chat_id, MSG_ERROR_INTERNAL + " (template_file is missing)")
        cmd_start(msg) 
        return

    try:
        relative_tpl_path = current_user_state["template_file"]
        tpl_path = os.path.join(templates_dir, relative_tpl_path)

        if not os.path.exists(tpl_path):
            bot.send_message(chat_id, MSG_TEMPLATE_NOT_FOUND)
            logging.warning(f"Шаблон не найден при обработке | chat={chat_id} path={tpl_path}")
            user_state[chat_id] = {}
            cmd_start(msg)
            return

        logging.debug(f"Шаблон выбран | chat={chat_id} path={tpl_path}")

        if msg.content_type == "photo":
            file_id = msg.photo[-1].file_id
        elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/"): # Добавлена проверка msg.document
            file_id = msg.document.file_id
        else:
            bot.reply_to(msg, "Пожалуйста, отправьте изображение как фото или как документ-картинку.")
            return
            
        file_info = bot.get_file(file_id)
        downloaded_file_bytes = bot.download_file(file_info.file_path)
        user_img = Image.open(io.BytesIO(downloaded_file_bytes))

        logging.debug(f"Изображение пользователя загружено | chat={chat_id} size={user_img.size}")

        processing_msg = bot.send_message(chat_id, MSG_PROCESSING)

        template_img = Image.open(tpl_path)
        result_bytes = process_template_photo(template_img, user_img)
        template_img.close()
        user_img.close()


        logging.info(f"Изображение обработано успешно | chat={chat_id} result_size={len(result_bytes)} bytes")

        kb = InlineKeyboardMarkup()
        kb.add(InlineKeyboardButton(BTN_REGENERATE, callback_data="start_over"))

        try:
            bot.delete_message(chat_id, processing_msg.message_id)
        except Exception as e:
            logging.warning(f"Не удалось удалить сообщение о обработке: {e}")

        if len(result_bytes) <= TG_PHOTO_LIMIT:
            bot.send_photo(chat_id, result_bytes, caption=MSG_DONE, reply_markup=kb)
        else:
            output_filename = "result.jpg" 
            if result_bytes.startswith(b'\x89PNG\r\n\x1a\n'): 
                 output_filename = "result.png"
            bot.send_document(chat_id, (output_filename, result_bytes),
                              caption=MSG_DONE, reply_markup=kb)

        user_state[chat_id] = {"persona": current_user_state.get("persona")} 

    except Exception as e:
        logging.exception(f"Ошибка при обработке медиа | chat={chat_id}")
        bot.send_message(chat_id, MSG_ERROR_INTERNAL)
        kb_error = InlineKeyboardMarkup()
        kb_error.add(InlineKeyboardButton("Начать заново", callback_data="start_over"))
        bot.send_message(chat_id, "Попробуйте начать заново.", reply_markup=kb_error)


# ────────────────────────────────────────────────────────────────────
# ► 10. ТОЧКА ВХОДА
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not BOT_TOKEN:
        print("Переменная окружения BOT_TOKEN не установлена.")
    else:
        logging.info("Запуск бота...")
        try:
            bot.remove_webhook()
            bot.infinity_polling(skip_pending=True) 
        except Exception as e:
            logging.exception("Неожиданная ошибка в главном цикле бота")
            sys.exit(1)
