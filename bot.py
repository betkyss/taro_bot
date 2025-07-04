#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram-бот для вставки НЕСКОЛЬКИХ фотографий пользователя в шаблон.
"""

# ────────────────────────────────────────────────────────────────────
# ► 0. ИМПОРТЫ
# ────────────────────────────────────────────────────────────────────
import io, os, math, random, traceback, warnings, sys, logging, time, re
from typing import Dict, Any, List, Tuple, Optional

import requests

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
        logging.StreamHandler(sys.stdout),
    ],
)


import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, Message

# ────────────────────────────────────────────────────────────────────
# ► 1. НАСТРОЙКА PIL
# ────────────────────────────────────────────────────────────────────
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# ────────────────────────────────────────────────────────────────────
# ► 2. ГЛОБАЛЬНЫЕ КОНСТАНТЫ
# ────────────────────────────────────────────────────────────────────
templates_dir = "templates"
filter_path = "filter.png"
OUT_DIM = 2048
scale_pixels = 75
upscale_factor = 0.5
SCALE_MONO = 8
min_shift, max_shift = 2, 8
min_rotation, max_rotation = 1, 3
thickness, box_blur_radius = 25, 5
MAX_PIXELS_TPL = 80_000_000
TG_PHOTO_LIMIT = 10_485_760
SEND_RETRIES = 3
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MIN_CONTOUR_AREA = 1000  # Минимальная площадь зеленой области для учета

# ────────────────────────────────────────────────────────────────────
# ► 3. ГОТОВЫЕ СООБЩЕНИЯ / ЭМОДЗИ
# ────────────────────────────────────────────────────────────────────
MSG_SELECT_PERSONA = "🎭 Выберите персонажа:"
MSG_SELECT_STAGE = "📟 Выберите этап"
MSG_SELECT_TEMPLATE = "🖼️ Выберите шаблон"
MSG_SELECT_VARIANT = "🔢 Выберите вариант"
MSG_SEND_PHOTO = "📥 Отправь фото {current_num} из {total_num}"  # ИЗМЕНЕНО
MSG_PROCESSING = "⏳ Идёт обработка…"
MSG_DONE = "✅ Готово!"
BTN_REGENERATE = "Сгенерировать снова"
MSG_NO_ACCESS = "Нет доступа."
MSG_START_FIRST = "Сначала /start."
MSG_TEMPLATE_NOT_FOUND = "Шаблон не найден."
MSG_ERROR_INTERNAL = "⚠️ Внутренняя ошибка. Посмотрите логи."
MSG_NO_TEMPLATES_FOUND = "Не найдено подходящих шаблонов для этого выбора."

STAGE_NAME_MAP = {
    "0": "0️⃣ (бесплатный расклад)",
    "1": "1️⃣ (рез-т платной д-ки)",
    "2": "2️⃣ (рез-т после ритуала)",
    "3": "3️⃣ (рез-т после ритуала)",
}


# ────────────────────────────────────────────────────────────────────
# ► 4. ЗАГРУЗКА «БЕЛОГО» СПИСКА ПОЛЬЗОВАТЕЛЕЙ
# ────────────────────────────────────────────────────────────────────
def load_allowed_user_ids(fname: str = "allowed_users.txt") -> set[int]:
    ids = set()
    if os.path.exists(fname):
        with open(fname, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        ids.add(int(line))
                    except ValueError:
                        logging.warning(
                            f"Не удалось прочитать User ID: {line} из {fname}"
                        )
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

import telebot.apihelper as tbh

tbh.SEND_FILE_TIMEOUT = 120
tbh.CONNECT_TIMEOUT = 30
tbh.READ_TIMEOUT = 120
logging.info("Bot ready")


# ────────────────────────────────────────────────────────────────────
# ► 6. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ────────────────────────────────────────────────────────────────────
def get_personas() -> list[str]:
    if not os.path.exists(templates_dir) or not os.path.isdir(templates_dir):
        logging.error(f"Папка шаблонов '{templates_dir}' не найдена.")
        return []
    return sorted(
        d
        for d in os.listdir(templates_dir)
        if os.path.isdir(os.path.join(templates_dir, d))
    )


def get_stages_for_persona(persona: str) -> List[Tuple[str, str]]:
    persona_path = os.path.join(templates_dir, persona)
    stages = []
    if os.path.isdir(persona_path):
        for item in os.listdir(persona_path):
            item_path = os.path.join(persona_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                display_name = STAGE_NAME_MAP.get(item, f"Этап {item}")
                stages.append((item, display_name))
    return sorted(stages, key=lambda x: int(x[0]))


def get_templates_in_path(current_path: str) -> List[str]:
    templates = []
    if os.path.isdir(current_path):
        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            if os.path.isfile(item_path) and item.lower().endswith(
                VALID_IMAGE_EXTENSIONS
            ):
                templates.append(item)
    return sorted(templates)


def group_templates_by_basename(current_path: str) -> Dict[str, Dict[int, str]]:
    """\
    Возвращает словарь групп шаблонов вида {basename: {number: filename}}.

    basename -- имя файла без суффикса _1/_2 и расширения. number -- 1, 2, ...
    """
    groups: Dict[str, Dict[int, str]] = {}
    for fname in get_templates_in_path(current_path):
        base = get_template_name_without_extension(fname)
        if base.endswith("_1") or base.endswith("_2"):
            key = base[:-2]
            try:
                num = int(base[-1])
            except ValueError:
                num = 1
        else:
            key = base
            num = 1
        groups.setdefault(key, {})[num] = fname
    return groups


def get_template_name_without_extension(filename: str) -> str:
    return os.path.splitext(filename)[0]


def order_corners(pts: np.ndarray) -> np.ndarray:
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


def _safe_send(func, *args, **kwargs):
    """Отправляет медиа с несколькими попытками при таймауте."""
    for attempt in range(1, SEND_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ReadTimeout:
            logging.warning(f"Timeout при отправке (попытка {attempt}/{SEND_RETRIES})")
            if attempt == SEND_RETRIES:
                raise
            time.sleep(2)


# ────────────────────────────────────────────────────────────────────
# ► 7. НОВЫЕ ФУНКЦИИ АНАЛИЗА И ОБРАБОТКИ
# ────────────────────────────────────────────────────────────────────


def find_and_sort_green_areas(tpl_img: Image.Image) -> List[np.ndarray]:
    """
    Находит все зеленые области в шаблоне, фильтрует слишком маленькие
    и сортирует их слева направо по X-координате центра.
    """
    tpl_rgba = tpl_img.convert("RGBA")
    b, g, r, _ = np.asarray(tpl_rgba).transpose(2, 0, 1)
    mask = ((g > 200) & (r < 100) & (b < 100)).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтруем контуры по минимальной площади
    significant_contours = [
        cnt for cnt in cnts if cv2.contourArea(cnt) > MIN_CONTOUR_AREA
    ]

    if not significant_contours:
        return []

    # Сортируем контуры по X-координате их ограничивающего прямоугольника
    # Это гарантирует обработку слева направо
    sorted_contours = sorted(significant_contours, key=lambda c: cv2.boundingRect(c)[0])

    return sorted_contours


def process_template_with_multiple_photos(
    tpl_img: Image.Image, user_imgs: List[Image.Image]
) -> bytes:
    """
    Основная функция обработки: находит зеленые области, сортирует их
    и вставляет в них фото из списка user_imgs.
    """
    # 7.1 --- АПСКЕЙЛ ШАБЛОНА
    out_scale = OUT_DIM / max(tpl_img.size) if max(tpl_img.size) < OUT_DIM else 1.0
    tpl_big = tpl_img.resize(
        (int(tpl_img.width * out_scale), int(tpl_img.height * out_scale)), Image.LANCZOS
    )
    if tpl_big.width * tpl_big.height > MAX_PIXELS_TPL:
        factor = math.sqrt(MAX_PIXELS_TPL / (tpl_big.width * tpl_big.height))
        tpl_big = tpl_big.resize(
            (int(tpl_big.width * factor), int(tpl_big.height * factor)), Image.LANCZOS
        )
        out_scale *= factor

    res = tpl_big.convert("RGBA")

    # 7.2 --- ПОИСК И СОРТИРОВКА ЗЕЛЕНЫХ ОБЛАСТЕЙ
    # Используем оригинальный, не масштабированный шаблон для поиска
    sorted_contours = find_and_sort_green_areas(tpl_img)

    if not sorted_contours:
        return _save_png(res)

    # Итерируемся по отсортированным контурам и предоставленным фото
    # min(len(..)) для безопасности, если фото прислали меньше чем надо
    for i in range(min(len(sorted_contours), len(user_imgs))):
        cnt = sorted_contours[i]
        user_img = user_imgs[i]

        # Масштабируем контур в соответствии с апскейлом шаблона
        scaled_cnt = (cnt * out_scale).astype(np.int32)

        # 7.3 --- определяем минимальный ограничивающий прямоугольник
        (cx, cy), (w_rect, h_rect), ang_rect = cv2.minAreaRect(scaled_cnt)
        long_side, short_side = int(max(w_rect, h_rect)), int(min(w_rect, h_rect))

        if long_side == 0 or short_side == 0:
            continue  # Пропускаем вырожденные области

        # 7.4 --- СОЗДАЕМ «МОНОЛИТ»
        H = int((long_side + scale_pixels * out_scale) * upscale_factor) * SCALE_MONO
        W = int((short_side + scale_pixels * out_scale) * upscale_factor) * SCALE_MONO
        if W == 0 or H == 0:
            continue

        usr_big = user_img.convert("RGBA").resize(
            (
                int(user_img.width * upscale_factor * SCALE_MONO),
                int(user_img.height * upscale_factor * SCALE_MONO),
            ),
            Image.LANCZOS,
        )
        sc_fill = max(W / usr_big.width, H / usr_big.height)
        usr_fill = usr_big.resize(
            (int(usr_big.width * sc_fill), int(usr_big.height * sc_fill)),
            Image.BICUBIC if sc_fill > 1 else Image.LANCZOS,
        )
        lft, top = (usr_fill.width - W) // 2, (usr_fill.height - H) // 2
        cropped = usr_fill.crop((lft, top, lft + W, top + H))

        mono = cropped  # По умолчанию, если фильтра нет
        if os.path.exists(filter_path):
            filt = Image.open(filter_path).convert("RGBA").resize((W, H), Image.LANCZOS)
            mono = Image.new("RGBA", filt.size, (0, 0, 0, 0))
            mono.paste(cropped, (0, 0))
            mono = Image.alpha_composite(mono, filt)

        mono = mono.rotate(
            random.choice([-1, 1]) * random.uniform(min_rotation, max_rotation),
            expand=True,
            resample=Image.BICUBIC,
        )
        mono = mono.resize(
            (mono.width // SCALE_MONO, mono.height // SCALE_MONO), Image.LANCZOS
        )
        dx, dy = random.choice([-1, 1]) * random.randint(
            min_shift, max_shift
        ), random.choice([-1, 1]) * random.randint(min_shift, max_shift)

        # 7.5 --- ВСТАВКА «МОНОЛИТА» БЕЗ ИСКАЖЕНИЙ
        effective_angle = ang_rect
        if w_rect < h_rect:
            mono_rot = mono.rotate(
                -effective_angle, expand=True, resample=Image.BICUBIC
            )
        else:
            mono_rot = mono.rotate(
                -effective_angle - 90, expand=True, resample=Image.BICUBIC
            )
        layer = Image.new("RGBA", res.size, (0, 0, 0, 0))
        paste_x, paste_y = int(cx + dx - mono_rot.width / 2), int(
            cy + dy - mono_rot.height / 2
        )
        layer.paste(mono_rot, (paste_x, paste_y), mono_rot)
        res = Image.alpha_composite(res, layer)

    # 7.6 --- РАЗМЫТИЕ ЛЕВОЙ КРОМКИ (применяется ко всему итоговому изображению)
    if thickness > 0 and box_blur_radius > 0 and res.width >= thickness:
        strip = res.crop((0, 0, thickness, res.height))
        res.paste(strip.filter(ImageFilter.BoxBlur(box_blur_radius)), (0, 0))

    # 7.7 --- ПАКОВКА
    png_bytes = _save_png(res)
    if len(png_bytes) <= TG_PHOTO_LIMIT:
        return png_bytes
    for q in (95, 90, 85, 80, 75, 70, 65):
        jpg_bytes = _save_jpeg(res, q)
        if len(jpg_bytes) <= TG_PHOTO_LIMIT:
            return jpg_bytes
    return _save_jpeg(res, 50)


# ────────────────────────────────────────────────────────────────────
# ► 8. ХРАНИЛИЩЕ СОСТОЯНИЙ
# ────────────────────────────────────────────────────────────────────
user_state: Dict[int, Dict[str, Any]] = {}


# ────────────────────────────────────────────────────────────────────
# ► 9. ХЕНДЛЕРЫ БОТА
# ────────────────────────────────────────────────────────────────────
def common_access_check(user_id: int, chat_id: int) -> bool:
    if user_id not in ALLOWED_USER_IDS:
        bot.send_message(chat_id, MSG_NO_ACCESS)
        return False
    return True


def common_access_check_callback(call) -> bool:
    if call.from_user.id not in ALLOWED_USER_IDS:
        bot.answer_callback_query(call.id, MSG_NO_ACCESS, show_alert=True)
        return False
    return True


def get_display_template_name(tpl_file_name: str, persona_name_str: str) -> str:
    base_name = get_template_name_without_extension(tpl_file_name)
    prefix_to_remove = f"{persona_name_str}_"
    display_name = (
        base_name[len(prefix_to_remove) :]
        if base_name.startswith(prefix_to_remove)
        else base_name
    )
    display_name = re.sub(r"_[0-9]+$", "", display_name)
    return display_name.capitalize()


def request_next_photo(chat_id: int, message_id: Optional[int] = None):
    """Запрашивает следующее фото или запускает обработку, если все собраны."""
    state = user_state.get(chat_id)
    if not state or state.get("state") != "waiting_photos":
        return

    required = state.get("required_photos", 0)
    photos = state.get("photos", [])

    if len(photos) < required:
        # Запрашиваем следующее фото
        msg = MSG_SEND_PHOTO.format(current_num=len(photos) + 1, total_num=required)
        if message_id:
            bot.edit_message_text(msg, chat_id, message_id)
        else:
            bot.send_message(chat_id, msg)
    else:
        # Все фото собраны, запускаем обработку
        processing_msg = bot.send_message(chat_id, MSG_PROCESSING)

        try:
            relative_tpl_path = state["template_file"]
            tpl_path = os.path.join(templates_dir, relative_tpl_path)

            template_img = Image.open(tpl_path)
            user_imgs = [Image.open(io.BytesIO(p_bytes)) for p_bytes in photos]

            result_bytes = process_template_with_multiple_photos(
                template_img, user_imgs
            )

            template_img.close()
            for img in user_imgs:
                img.close()

            logging.info(
                f"Изображение обработано успешно | chat={chat_id} result_size={len(result_bytes)} bytes"
            )
            kb = InlineKeyboardMarkup().add(
                InlineKeyboardButton(BTN_REGENERATE, callback_data="start_over")
            )

            try:
                bot.delete_message(chat_id, processing_msg.message_id)
            except Exception:
                pass

            if len(result_bytes) <= TG_PHOTO_LIMIT:
                _safe_send(
                    bot.send_photo,
                    chat_id,
                    result_bytes,
                    caption=MSG_DONE,
                    reply_markup=kb,
                )
            else:
                fname = (
                    "result.png"
                    if result_bytes.startswith(b"\x89PNG")
                    else "result.jpg"
                )
                _safe_send(
                    bot.send_document,
                    chat_id,
                    (fname, result_bytes),
                    caption=MSG_DONE,
                    reply_markup=kb,
                )

            # Сбрасываем состояние для этого пользователя
            user_state[chat_id] = {}

        except Exception as e:
            logging.exception(f"Ошибка при финальной обработке медиа | chat={chat_id}")
            bot.send_message(chat_id, MSG_ERROR_INTERNAL)
            kb_error = InlineKeyboardMarkup().add(
                InlineKeyboardButton("Начать заново", callback_data="start_over")
            )
            bot.send_message(
                chat_id, "Попробуйте начать заново.", reply_markup=kb_error
            )


@bot.callback_query_handler(func=lambda c: c.data == "start_over")
def cb_start_again(call) -> None:
    if not common_access_check_callback(call):
        return
    bot.answer_callback_query(call.id)
    cmd_start(call.message, from_callback=True)


@bot.message_handler(commands=["start"])
def cmd_start(msg: Message, from_callback: bool = False) -> None:
    if not from_callback and not common_access_check(msg.from_user.id, msg.chat.id):
        return
    chat_id = msg.chat.id
    user_state[chat_id] = {}
    personas = get_personas()
    if not personas:
        bot.send_message(chat_id, "Персонажи не найдены.")
        return
    kb = InlineKeyboardMarkup()
    for p_name in personas:
        kb.add(InlineKeyboardButton(p_name, callback_data=f"persona_{p_name}"))

    if from_callback:
        try:
            bot.edit_message_text(
                MSG_SELECT_PERSONA, chat_id, msg.message_id, reply_markup=kb
            )
        except Exception as e:
            bot.send_message(chat_id, MSG_SELECT_PERSONA, reply_markup=kb)
    else:
        bot.send_message(chat_id, MSG_SELECT_PERSONA, reply_markup=kb)


def set_template_and_start_photo_collection(
    chat_id: int, message_id: int, template_relative_path: str
):
    """
    Анализирует шаблон, обновляет состояние и запрашивает первое фото.
    """
    full_tpl_path = os.path.join(templates_dir, template_relative_path)
    if not os.path.exists(full_tpl_path):
        bot.edit_message_text(MSG_TEMPLATE_NOT_FOUND, chat_id, message_id)
        return

    try:
        with Image.open(full_tpl_path) as tpl_img:
            # Анализируем, сколько областей в шаблоне
            num_areas = len(find_and_sort_green_areas(tpl_img))
    except Exception as e:
        logging.error(f"Не удалось проанализировать шаблон {full_tpl_path}: {e}")
        bot.edit_message_text(
            "Не удалось обработать файл шаблона.", chat_id, message_id
        )
        return

    if num_areas == 0:
        bot.edit_message_text(
            "В этом шаблоне не найдено областей для вставки.", chat_id, message_id
        )
        return

    # Обновляем состояние пользователя
    user_state[chat_id]["template_file"] = template_relative_path
    user_state[chat_id]["state"] = "waiting_photos"  # Новое состояние
    user_state[chat_id]["required_photos"] = num_areas
    user_state[chat_id]["photos"] = []  # Список для хранения байтов фото

    # Запрашиваем первое фото
    request_next_photo(chat_id, message_id=message_id)


def _show_variant_options(
    chat_id: int, message_id: int, rel_dir: str, variants: Dict[int, str]
):
    """Показывает выбор между вариантами _1/_2, если их несколько."""
    kb = InlineKeyboardMarkup()
    for num, fname in sorted(variants.items()):
        label = f"{num} фото" if num in (1, 2) else str(num)
        kb.add(InlineKeyboardButton(label, callback_data=f"template_{rel_dir}/{fname}"))
    bot.edit_message_text(MSG_SELECT_VARIANT, chat_id, message_id, reply_markup=kb)


def _handle_template_groups(
    chat_id: int,
    message_id: int,
    persona_name: str,
    stage_val: Optional[str],
    current_path: str,
):
    """Обрабатывает группировку шаблонов и вывод выбора пользователю."""
    groups = group_templates_by_basename(current_path)
    if not groups:
        bot.edit_message_text(MSG_NO_TEMPLATES_FOUND, chat_id, message_id)
        return

    rel_dir = os.path.join(persona_name, stage_val) if stage_val else persona_name

    if len(groups) == 1:
        base, variants = next(iter(groups.items()))
        if len(variants) == 1:
            file = next(iter(variants.values()))
            template_path = os.path.join(rel_dir, file)
            set_template_and_start_photo_collection(chat_id, message_id, template_path)
        else:
            _show_variant_options(chat_id, message_id, rel_dir, variants)
    else:
        kb = InlineKeyboardMarkup()
        for base, variants in sorted(groups.items()):
            display_name = get_display_template_name(
                list(variants.values())[0], persona_name
            )
            kb.add(
                InlineKeyboardButton(
                    display_name, callback_data=f"tplgrp_{rel_dir}/{base}"
                )
            )
        bot.edit_message_text(MSG_SELECT_TEMPLATE, chat_id, message_id, reply_markup=kb)


@bot.callback_query_handler(func=lambda c: c.data.startswith("persona_"))
def cb_persona(call) -> None:
    if not common_access_check_callback(call):
        return
    chat_id = call.message.chat.id
    persona_name = call.data.split("_", 1)[1]
    user_state[chat_id] = {"persona": persona_name}
    bot.answer_callback_query(call.id)
    # ... (логика выбора этапа или шаблона) ...
    # Эта часть остается похожей, но в конце вызывает set_template_and_start_photo_collection
    persona_path = os.path.join(templates_dir, persona_name)
    stages = get_stages_for_persona(persona_name)

    if stages:
        kb = InlineKeyboardMarkup()
        for stage_val, stage_label in stages:
            kb.add(
                InlineKeyboardButton(stage_label, callback_data=f"stage_{stage_val}")
            )
        bot.edit_message_text(
            MSG_SELECT_STAGE, chat_id, call.message.message_id, reply_markup=kb
        )
    else:  # Нет этапов, сразу шаблоны
        _handle_template_groups(
            chat_id, call.message.message_id, persona_name, None, persona_path
        )


@bot.callback_query_handler(func=lambda c: c.data.startswith("stage_"))
def cb_stage(call) -> None:
    if not common_access_check_callback(call):
        return
    chat_id = call.message.chat.id
    stage_val = call.data.split("_", 1)[1]
    if "persona" not in user_state.get(chat_id, {}):
        cmd_start(call.message, from_callback=True)
        return
    user_state[chat_id]["stage"] = stage_val
    bot.answer_callback_query(call.id)
    persona_name = user_state[chat_id]["persona"]
    current_path = os.path.join(templates_dir, persona_name, stage_val)
    _handle_template_groups(
        chat_id, call.message.message_id, persona_name, stage_val, current_path
    )


@bot.callback_query_handler(func=lambda c: c.data.startswith("tplgrp_"))
def cb_template_group(call) -> None:
    if not common_access_check_callback(call):
        return
    chat_id = call.message.chat.id
    data = call.data.split("_", 1)[1]
    parts = data.split("/")
    base = parts[-1]
    rel_dir = "/".join(parts[:-1])
    persona_name = user_state.get(chat_id, {}).get("persona")
    if not persona_name:
        cmd_start(call.message, from_callback=True)
        return
    stage_val = user_state.get(chat_id, {}).get("stage") if len(parts) > 2 else None
    current_path = os.path.join(templates_dir, rel_dir)
    groups = group_templates_by_basename(current_path)
    variants = groups.get(base)
    bot.answer_callback_query(call.id)
    if not variants:
        bot.edit_message_text(MSG_TEMPLATE_NOT_FOUND, chat_id, call.message.message_id)
        return
    if len(variants) == 1:
        file = next(iter(variants.values()))
        template_path = os.path.join(rel_dir, file)
        set_template_and_start_photo_collection(
            chat_id, call.message.message_id, template_path
        )
    else:
        _show_variant_options(chat_id, call.message.message_id, rel_dir, variants)


@bot.callback_query_handler(func=lambda c: c.data.startswith("template_"))
def cb_template_selection(call) -> None:
    if not common_access_check_callback(call):
        return
    chat_id = call.message.chat.id
    template_relative_path = call.data.split("_", 1)[1]
    if "persona" not in user_state.get(chat_id, {}):
        cmd_start(call.message, from_callback=True)
        return
    bot.answer_callback_query(call.id)
    set_template_and_start_photo_collection(
        chat_id, call.message.message_id, template_relative_path
    )


@bot.message_handler(content_types=["photo", "document"])
def handle_media(msg: Message) -> None:
    chat_id = msg.chat.id
    if not common_access_check(msg.from_user.id, chat_id):
        return

    state = user_state.get(chat_id, {})
    if state.get("state") != "waiting_photos":
        bot.send_message(chat_id, MSG_START_FIRST)
        return

    try:
        if msg.content_type == "photo":
            file_id = msg.photo[-1].file_id
        elif (
            msg.document
            and msg.document.mime_type
            and msg.document.mime_type.startswith("image/")
        ):
            file_id = msg.document.file_id
        else:
            bot.reply_to(msg, "Пожалуйста, отправьте изображение.")
            return

        file_info = bot.get_file(file_id)
        downloaded_file_bytes = bot.download_file(file_info.file_path)

        # Добавляем байты фото в состояние
        user_state[chat_id]["photos"].append(downloaded_file_bytes)
        logging.debug(
            f"Фото {len(user_state[chat_id]['photos'])}/{user_state[chat_id]['required_photos']} получено | chat={chat_id}"
        )

        # Удаляем сообщение с просьбой прислать фото
        try:
            bot.delete_message(chat_id, msg.message_id)
        except Exception:
            pass

        # Запрашиваем следующее фото или запускаем обработку
        request_next_photo(chat_id)

    except Exception as e:
        logging.exception(f"Ошибка при обработке медиа | chat={chat_id}")
        bot.send_message(chat_id, MSG_ERROR_INTERNAL)


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
