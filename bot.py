#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io, os, math, random, traceback, warnings

import numpy as np
import cv2
from PIL import Image, ImageFilter
from dotenv import load_dotenv
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, Message

# ───── СБРОС ЛИМИТА PIL И ПОДАВЛЕНИЕ ВАРНИНГА ─────
Image.MAX_IMAGE_PIXELS = None                                   # убираем «бомбу»
warnings.simplefilter('ignore', Image.DecompressionBombWarning) # скрываем предупреждение

# ───── КОНСТАНТЫ ─────
templates_dir    = "templates"
filter_path      = "filter_2.png"
OUT_DIM          = 4096          # итог. длинная сторона
scale_pixels     = 60
upscale_factor   = 3.0
SCALE_MONO       = 3
min_shift, max_shift       = 5, 15
min_rotation, max_rotation = 2, 4
thickness, box_blur_radius = 25, 5
MAX_PIXELS_TPL   = 80_000_000    # 80 Мп: потолок для upscaled-шаблона
TG_PHOTO_LIMIT   = 10_485_760    # 10 МБ

# ───── 1. белый список ─────
def load_allowed_user_ids(fname="allowed_users.txt"):
    ids=set()
    if os.path.exists(fname):
        with open(fname, encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line and not line.startswith("#"):
                    try: ids.add(int(line))
                    except: pass
    print(f"[DEBUG] белый список: {len(ids)} id")
    return ids
ALLOWED_USER_IDS = load_allowed_user_ids()

# ───── 2. бот ─────
load_dotenv()
bot = telebot.TeleBot(os.getenv("BOT_TOKEN"))
import telebot.apihelper as tbh
tbh.SEND_FILE_TIMEOUT, tbh.CONNECT_TIMEOUT, tbh.READ_TIMEOUT = 120, 30, 30
print("Bot ready")

# ───── 3. helpers ─────
def get_personas():
    return sorted(d for d in os.listdir(templates_dir)
                  if os.path.isdir(os.path.join(templates_dir, d)))

def build_persona_actions_mapping():
    mp={}
    for p in get_personas():
        if p=="JUL": continue
        folder=os.path.join(templates_dir,p)
        acts={}
        for f in os.listdir(folder):
            name,ext=os.path.splitext(f)
            if ext.lower() not in(".jpg",".jpeg",".png") or "_" not in name: continue
            _,act=name.split("_",1)
            acts[act]=f
        if acts: mp[p]=acts
    return mp

def order_corners(pts):
    pts=np.array(pts,dtype="float32")
    s=pts.sum(1); diff=np.diff(pts,axis=1)
    tl,br=pts[np.argmin(s)],pts[np.argmax(s)]
    tr,bl=pts[np.argmin(diff)],pts[np.argmax(diff)]
    return np.array([tl,tr,br,bl],dtype="float32")

def _save_png(img:Image.Image)->bytes:
    b=io.BytesIO(); img.save(b,"PNG"); b.seek(0); return b.getvalue()
def _save_jpeg(img:Image.Image,q=95)->bytes:
    b=io.BytesIO(); img.convert("RGB").save(b,"JPEG",quality=q); b.seek(0); return b.getvalue()

# ───── 4. обработка изображения ─────
def process_template_photo(tpl_img:Image.Image, user_img:Image.Image)->bytes:
    # 4.1 масштаб шаблона → OUT_DIM
    out_scale = OUT_DIM / max(tpl_img.size) if max(tpl_img.size) < OUT_DIM else 1.0
    tpl_big   = tpl_img.resize((int(tpl_img.width*out_scale),
                                int(tpl_img.height*out_scale)), Image.LANCZOS)

    # 4.1a ограничение 80 Мп (чтобы не взорвать RAM)
    if tpl_big.width * tpl_big.height > MAX_PIXELS_TPL:
        factor = math.sqrt(MAX_PIXELS_TPL / (tpl_big.width * tpl_big.height))
        tpl_big = tpl_big.resize((int(tpl_big.width*factor), int(tpl_big.height*factor)),
                                 Image.LANCZOS)
        out_scale *= factor

    tpl_rgba = tpl_big.convert("RGBA")

    # 4.2 маска зелёной рамки
    b,g,r,_ = np.asarray(tpl_rgba).transpose(2,0,1)
    mask = ((g>200)&(r<100)&(b<100)).astype(np.uint8)*255
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return _save_png(tpl_rgba)
    cnt = max(cnts, key=cv2.contourArea)

    # 4.3 четырёхугольник или minAreaRect
    peri  = cv2.arcLength(cnt,True)
    approx= cv2.approxPolyDP(cnt,0.02*peri,True)
    persp = len(approx)==4
    if persp:
        quad = order_corners([p[0] for p in approx])
        center = quad.mean(0,keepdims=True)
        vecs   = quad-center; lens=np.linalg.norm(vecs,1,keepdims=True)
        quad  += vecs/(lens+1e-6)*scale_pixels*out_scale
        (wA,hA)=(np.linalg.norm(quad[0]-quad[1]),np.linalg.norm(quad[0]-quad[3]))
        (wB,hB)=(np.linalg.norm(quad[2]-quad[3]),np.linalg.norm(quad[1]-quad[2]))
        long_side, short_side = int(max(hA,hB)), int(max(wA,wB))
    else:
        rect=cv2.minAreaRect(cnt); ((cx0,cy0),(w0,h0),ang)=rect
        long_side, short_side = int(max(w0,h0)), int(min(w0,h0))

    # 4.4 создаём монолит крупнее (×SCALE_MONO)
    H = int((long_side + scale_pixels*out_scale) * upscale_factor) * SCALE_MONO
    W = int((short_side+ scale_pixels*out_scale) * upscale_factor) * SCALE_MONO

    usr_big = user_img.convert("RGBA").resize(
        (int(user_img.width  * upscale_factor * SCALE_MONO),
         int(user_img.height * upscale_factor * SCALE_MONO)), Image.LANCZOS)
    sc = max(W/usr_big.width, H/usr_big.height)
    usr_fit=usr_big.resize((int(usr_big.width*sc), int(usr_big.height*sc)), Image.LANCZOS)
    left=(usr_fit.width-W)//2; top=(usr_fit.height-H)//2
    cropped = usr_fit.crop((left,top,left+W,top+H))

    filt = Image.open(filter_path).convert("RGBA").resize((W,H+30), Image.LANCZOS)
    mono = Image.new("RGBA", filt.size,(0,0,0,0))
    mono.paste(cropped,(0,0),cropped)
    mono = Image.alpha_composite(mono,filt)

    mono = mono.rotate(random.choice([-1,1])*random.uniform(min_rotation,max_rotation),
                       expand=True, resample=Image.BICUBIC)
    mono = mono.resize((mono.width//SCALE_MONO, mono.height//SCALE_MONO), Image.LANCZOS)

    dx = random.choice([-1,1])*random.randint(min_shift,max_shift)
    dy = random.choice([-1,1])*random.randint(min_shift,max_shift)

    if persp:
        src = np.array([[0,0],[W//SCALE_MONO,0],
                        [W//SCALE_MONO,(H+30)//SCALE_MONO],[0,(H+30)//SCALE_MONO]], dtype="float32")
        quad_shift = quad + np.array([dx,dy],dtype="float32")
        M = cv2.getPerspectiveTransform(src, quad_shift)

        canvas_bgr = cv2.cvtColor(np.asarray(tpl_rgba), cv2.COLOR_RGBA2BGRA)
        mono_bgr   = cv2.cvtColor(np.asarray(mono),    cv2.COLOR_RGBA2BGRA)
        warp = cv2.warpPerspective(mono_bgr, M, dsize=tpl_rgba.size,
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,0,0,0))
        a = warp[:,:,3:4]/255.0
        canvas_bgr[:,:,:3] = canvas_bgr[:,:,:3]*(1-a) + warp[:,:,:3]*a
        res = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGRA2RGBA), "RGBA")
    else:
        rect=cv2.minAreaRect(cnt); ((cx,cy),(w0,h0),ang)=rect
        ang_long = ang + (90 if w0<h0 else 0)
        rot = ang_long-90
        if rot>90: rot-=180
        elif rot<-90: rot+=180

        mono_rot = mono.rotate(-rot, expand=True, resample=Image.BICUBIC)
        layer = Image.new("RGBA", tpl_rgba.size,(0,0,0,0))
        layer.paste(mono_rot,(int(cx+dx-mono_rot.width/2),
                              int(cy+dy-mono_rot.height/2)), mono_rot)
        res = Image.alpha_composite(tpl_rgba, layer)

    # 4.5 размытие левого края
    if thickness and box_blur_radius:
        strip = res.crop((0,0,thickness,res.height))
        res.paste(strip.filter(ImageFilter.BoxBlur(box_blur_radius)), (0,0))

    # 4.6 упаковка под 10 МБ
    if max(res.size) <= OUT_DIM:
        png = _save_png(res)
        if len(png) <= TG_PHOTO_LIMIT:
            return png   # lossless
    # иначе JPEG с подбором качества
    for q in (95,90,85,80,75,70,65):
        jpg = _save_jpeg(res,q)
        if len(jpg) <= TG_PHOTO_LIMIT:
            return jpg
    return _save_jpeg(res,50)  # крайний случай

# ───── 5. Telegram-логика ─────
user_state={}
@bot.message_handler(commands=["start"])
def cmd_start(m:Message):
    if m.from_user.id not in ALLOWED_USER_IDS:
        bot.reply_to(m,"Нет доступа."); return
    kb=InlineKeyboardMarkup()
    for p in get_personas():
        kb.add(InlineKeyboardButton(p,callback_data=f"persona_{p}"))
    bot.send_message(m.chat.id,"Выберите персонажа:",reply_markup=kb)

@bot.callback_query_handler(func=lambda c:c.data.startswith("persona_"))
def cb_persona(c):
    if c.from_user.id not in ALLOWED_USER_IDS:
        bot.answer_callback_query(c.id,"Нет доступа."); return
    chat=c.message.chat.id; p=c.data.split("_",1)[1]
    user_state[chat]={"persona":p}
    if p=="JUL":
        kb=InlineKeyboardMarkup()
        for v,l in [("0","0 – бесплатный"),("1","1 – диагностика"),
                    ("2","2 – после ритуала"),("3","3 – после ритуала")]:
            kb.add(InlineKeyboardButton(l,callback_data=f"stage_{v}"))
        user_state[chat]["state"]="choose_stage"
        bot.send_message(chat,"Выберите этап:",reply_markup=kb)
    else:
        acts=sorted(build_persona_actions_mapping().get(p,{}).keys())
        kb=InlineKeyboardMarkup()
        for a in acts: kb.add(InlineKeyboardButton(a,callback_data=f"action_{a}"))
        user_state[chat]["state"]="choose_action"
        bot.send_message(chat,"Выберите шаблон:",reply_markup=kb)

@bot.callback_query_handler(func=lambda c:c.data.startswith("stage_"))
def cb_stage(c):
    st=c.data.split("_",1)[1]; chat=c.message.chat.id
    user_state[chat]["stage"]=st
    if st in("1","2"):
        user_state[chat]["state"]="waiting_photo"
        bot.send_message(chat,"Отправьте фото")
    else:
        folder=os.path.join(templates_dir,"JUL",st)
        try:
            files=[f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
        except FileNotFoundError:
            files=[]
        acts=sorted({os.path.splitext(f)[0].split("_",1)[1] for f in files})
        kb=InlineKeyboardMarkup()
        for a in acts: kb.add(InlineKeyboardButton(a,callback_data=f"action_{a}"))
        user_state[chat]["state"]="choose_action"
        bot.send_message(chat,"Выберите шаблон:",reply_markup=kb)

@bot.callback_query_handler(func=lambda c:c.data.startswith("action_"))
def cb_action(c):
    chat=c.message.chat.id
    user_state[chat]["action"]=c.data.split("_",1)[1]
    user_state[chat]["state"]="waiting_photo"
    bot.send_message(chat,"Отправьте фото")

@bot.message_handler(content_types=["photo","document"])
def handle_media(m:Message):
    chat=m.chat.id
    if user_state.get(chat,{}).get("state")!="waiting_photo":
        bot.send_message(chat,"Сначала /start."); return

    pers=user_state[chat]["persona"]; st=user_state[chat].get("stage"); act=user_state[chat].get("action")
    if pers=="JUL":
        if st in("1","2"):
            tpl_path=os.path.join(templates_dir,"JUL",st,"JUL.jpeg")
        else:
            base=os.path.join(templates_dir,"JUL",st,f"JUL_{act}")
            tpl_path=next((base+e for e in(".jpg",".jpeg",".png") if os.path.exists(base+e)),None)
    else:
        fname=build_persona_actions_mapping().get(pers,{}).get(act)
        tpl_path=os.path.join(templates_dir,pers,fname) if fname else None
    if not tpl_path or not os.path.exists(tpl_path):
        bot.send_message(chat,"Шаблон не найден."); return

    fid=m.photo[-1].file_id if m.content_type=="photo" else m.document.file_id
    try:
        data=bot.download_file(bot.get_file(fid).file_path)
        user_img=Image.open(io.BytesIO(data))
    except Exception:
        bot.send_message(chat,"Не удалось открыть фото."); return

    bot.send_message(chat,"Обработка...")
    try:
        result_bytes=process_template_photo(Image.open(tpl_path), user_img)
    except Exception:
        print("[ERR]",traceback.format_exc())
        bot.send_message(chat,"Ошибка обработки."); return

    kb=InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton("Сгенерировать снова",callback_data="persona_"+pers))

    # ── отправка: photo ≤10 МБ, иначе document
    if len(result_bytes) <= TG_PHOTO_LIMIT:
        bot.send_photo(chat, result_bytes, caption="Готово!", reply_markup=kb)
    else:
        bot.send_document(chat, ("result.png", result_bytes), caption="Готово!", reply_markup=kb)

    user_state[chat]["state"]=None

# ───── 6. run ─────
if __name__=="__main__":
    bot.remove_webhook()
    bot.infinity_polling()

