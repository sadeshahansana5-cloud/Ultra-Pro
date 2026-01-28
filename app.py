import os
import re
import time
import threading
from datetime import datetime, timezone

import requests
from flask import Flask
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError
from pyrogram import Client, filters
from pyrogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton, Message
)

# -------------------------
# ENV
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "").strip()

ALLOWED_GROUP_ID = int(os.getenv("ALLOWED_GROUP_ID", "0"))
ADMIN_REQ_CHANNEL_ID = int(os.getenv("ADMIN_REQ_CHANNEL_ID", "0"))

MONGO_URI = os.getenv("MONGO_URI", "").strip()

FILES_DB_NAME = os.getenv("FILES_DB_NAME", "autofilter").strip()
FILES_COLLECTION = os.getenv("FILES_COLLECTION", "royal_files").strip()

NEW_DB_NAME = os.getenv("NEW_DB_NAME", "file_search_bot").strip()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

RESULT_BUTTONS = int(os.getenv("RESULT_BUTTONS", "10"))
MAX_REQUESTS_PER_USER = int(os.getenv("MAX_REQUESTS_PER_USER", "3"))

PORT = int(os.getenv("PORT", "10000"))

# -------------------------
# BASIC CHECKS
# -------------------------
if not BOT_TOKEN or not API_HASH or API_ID == 0:
    raise RuntimeError("Missing BOT_TOKEN / API_ID / API_HASH env vars")
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI env var")
if not TMDB_API_KEY:
    raise RuntimeError("Missing TMDB_API_KEY env var")
if ALLOWED_GROUP_ID == 0 or ADMIN_REQ_CHANNEL_ID == 0:
    raise RuntimeError("Missing ALLOWED_GROUP_ID / ADMIN_REQ_CHANNEL_ID env vars")

# -------------------------
# Mongo
# -------------------------
mongo = MongoClient(MONGO_URI, connectTimeoutMS=10000, serverSelectionTimeoutMS=10000)

files_db = mongo[FILES_DB_NAME]
files_col = files_db[FILES_COLLECTION]

new_db = mongo[NEW_DB_NAME]
users_col = new_db["users"]
requests_col = new_db["requests"]

# Indexes (safe to call repeatedly)
users_col.create_index([("_id", ASCENDING)])
requests_col.create_index([("user_id", ASCENDING), ("status", ASCENDING), ("created_at", DESCENDING)])
requests_col.create_index([("media_type", ASCENDING), ("tmdb_id", ASCENDING), ("status", ASCENDING)])

# -------------------------
# Flask health server (Render needs port)
# -------------------------
app_web = Flask(__name__)

@app_web.get("/")
def home():
    return {"ok": True, "service": "movie-request-bot"}

def run_web():
    app_web.run(host="0.0.0.0", port=PORT)

# -------------------------
# Helpers: normalization for messy file_name
# -------------------------
GARBAGE_TOKENS = {
    "1080p","720p","480p","2160p","4k",
    "webrip","webdl","web-dl","bluray","brrip","hdrip","dvdrip",
    "x264","x265","h264","h265","hevc","aac","ddp","dd","dts",
    "10bit","8bit","hdr","sdr","proper","repack",
    "mkv","mp4","avi",
    "psa","yts","rarbg"
}

def normalize_title(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\.(mkv|mp4|avi)$", "", s)
    s = re.sub(r"[\[\]\(\)\{\}\|_]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    parts = [p for p in s.split() if p and p not in GARBAGE_TOKENS]
    return " ".join(parts).strip()

SEASON_EP_RE = re.compile(r"(s\d{1,2}e\d{1,2})", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

def guess_is_tv(query: str) -> bool:
    return bool(SEASON_EP_RE.search(query.lower()))

# -------------------------
# TMDB API
# -------------------------
TMDB_BASE = "https://api.themoviedb.org/3"

def tmdb_get(path: str, params: dict | None = None):
    params = params or {}
    params["api_key"] = TMDB_API_KEY
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def tmdb_search_multi(q: str):
    # search/multi returns movie + tv + person; we will filter movie/tv only
    data = tmdb_get("/search/multi", {"query": q, "include_adult": "false", "language": "en-US", "page": 1})
    out = []
    for it in data.get("results", []):
        mt = it.get("media_type")
        if mt not in ("movie", "tv"):
            continue
        out.append(it)
    return out

def tmdb_details(media_type: str, tmdb_id: int):
    if media_type == "movie":
        return tmdb_get(f"/movie/{tmdb_id}", {"language": "en-US"})
    return tmdb_get(f"/tv/{tmdb_id}", {"language": "en-US"})

# -------------------------
# Availability check in old DB
# (title based; fast enough for now. If needed later: store normalized field in new DB cache)
# -------------------------
def available_in_old_db(title: str) -> bool:
    norm = normalize_title(title)
    if not norm:
        return False
    # Use regex "contains" search over file_name. (Mongo can't use index well for contains.)
    # But with 20k docs it's OK. If later slow, we can add cache collection.
    try:
        q = {"file_name": {"$regex": re.escape(norm.split()[0]), "$options": "i"}}
        # quick prefilter by first token, then python-side verify contains
        cursor = files_col.find(q, {"file_name": 1}).limit(50)
        for d in cursor:
            fn = normalize_title(d.get("file_name", ""))
            if norm and norm in fn:
                return True
    except PyMongoError:
        return False
    return False

# -------------------------
# Request system
# -------------------------
def get_user_requests(user_id: int):
    return list(requests_col.find({"user_id": user_id, "status": "pending"}).sort("created_at", DESCENDING))

def can_add_request(user_id: int) -> bool:
    return requests_col.count_documents({"user_id": user_id, "status": "pending"}) < MAX_REQUESTS_PER_USER

def add_request(user_id: int, media_type: str, tmdb_id: int, title: str, year: str | None):
    doc = {
        "user_id": user_id,
        "media_type": media_type,
        "tmdb_id": tmdb_id,
        "title": title,
        "year": year,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
    }
    requests_col.insert_one(doc)
    return doc

def cancel_request(req_id):
    requests_col.update_one({"_id": req_id}, {"$set": {"status": "cancelled"}})

def mark_done_by_tmdb(media_type: str, tmdb_id: int):
    requests_col.update_many(
        {"media_type": media_type, "tmdb_id": tmdb_id, "status": "pending"},
        {"$set": {"status": "done", "done_at": datetime.now(timezone.utc)}}
    )

# -------------------------
# Telegram bot
# -------------------------
bot = Client(
    "file_search_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
)

def group_only_guard(_, __, m: Message):
    # allow PM for callbacks/details; restrict group features to allowed group
    if m.chat and m.chat.type in ("group", "supergroup"):
        return m.chat.id == ALLOWED_GROUP_ID
    return True

# -------------------------
# UI builders
# -------------------------
def build_results_keyboard(items):
    btns = []
    for it in items[:RESULT_BUTTONS]:
        mt = it.get("media_type")
        tid = it.get("id")
        title = it.get("title") if mt == "movie" else it.get("name")
        date = it.get("release_date") if mt == "movie" else it.get("first_air_date")
        year = date[:4] if date else "----"
        label = f"{'🎬' if mt=='movie' else '📺'} {title} ({year})"
        cb = f"det|{mt}|{tid}"
        btns.append([InlineKeyboardButton(label, callback_data=cb)])
    # start PM button
    btns.append([InlineKeyboardButton("🔔 Open bot in PM", url=f"https://t.me/{bot.me.username}?start=hi")])
    return InlineKeyboardMarkup(btns)

def build_detail_caption(det: dict, media_type: str, is_available: bool):
    if media_type == "movie":
        title = det.get("title") or "Unknown"
        date = det.get("release_date") or ""
        runtime = det.get("runtime")
        extra = f"⏱ Runtime: {runtime} min" if runtime else ""
    else:
        title = det.get("name") or "Unknown"
        date = det.get("first_air_date") or ""
        seasons = det.get("number_of_seasons")
        episodes = det.get("number_of_episodes")
        extra = f"📺 Seasons: {seasons} | Episodes: {episodes}" if seasons else ""

    year = date[:4] if date else "----"
    rating = det.get("vote_average")
    genres = ", ".join([g["name"] for g in det.get("genres", [])]) if det.get("genres") else "-"
    overview = det.get("overview") or "No overview."

    if is_available:
        avail_line = "✅ Available in our bot / අපේ බොට් එකේ තියෙනවා ✅"
    else:
        avail_line = "❌ Not available / අපේ බොට් එකේ නැහැ ❌"

    link = f"https://www.themoviedb.org/{'movie' if media_type=='movie' else 'tv'}/{det.get('id')}"

    cap = (
        f"**{title} ({year})**\n"
        f"⭐ Rating: `{rating}`\n"
        f"🎭 Genres: {genres}\n"
        f"{extra}\n\n"
        f"{overview}\n\n"
        f"{avail_line}\n"
        f"TMDB: {link}"
    )
    return cap

def build_detail_keyboard(media_type: str, tmdb_id: int, is_available: bool):
    rows = []
    if not is_available:
        rows.append([InlineKeyboardButton("📥 Request this", callback_data=f"req|{media_type}|{tmdb_id}")])
    return InlineKeyboardMarkup(rows) if rows else None

def build_replace_keyboard(reqs):
    rows = []
    for r in reqs[:MAX_REQUESTS_PER_USER]:
        t = r.get("title", "Unknown")
        y = r.get("year") or ""
        rid = str(r["_id"])
        rows.append([InlineKeyboardButton(f"🗑 Remove: {t} {y}".strip(), callback_data=f"rmreq|{rid}")])
    return InlineKeyboardMarkup(rows)

def admin_request_keyboard(media_type: str, tmdb_id: int, user_id: int):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Done", callback_data=f"adm_done|{media_type}|{tmdb_id}")],
        [InlineKeyboardButton("🗑 Cancel request", callback_data=f"adm_cancel|{media_type}|{tmdb_id}|{user_id}")]
    ])

# -------------------------
# Handlers
# -------------------------
@bot.on_message(filters.command("start"))
async def start(_, m: Message):
    # register/update user in new DB
    u = m.from_user
    if u:
        users_col.update_one(
            {"_id": u.id},
            {"$set": {
                "name": (u.first_name or "") + (" " + u.last_name if u.last_name else ""),
                "username": u.username,
                "last_seen": datetime.now(timezone.utc),
            }, "$setOnInsert": {"first_seen": datetime.now(timezone.utc)}},
            upsert=True
        )
    await m.reply_text("✅ Bot is active.\nGroup search works only in the allowed group.\nPM details will arrive here.")

@bot.on_message(filters.text & ~filters.private & filters.create(group_only_guard))
async def group_search(_, m: Message):
    q = (m.text or "").strip()
    if not q or q.startswith("/"):
        return

    # TMDB search
    try:
        results = tmdb_search_multi(q)
    except Exception:
        await m.reply_text("TMDB error. Try again later.")
        return

    if not results:
        await m.reply_text("No results found.")
        return

    kb = build_results_keyboard(results)
    await m.reply_text("Select a title:", reply_markup=kb)

@bot.on_message(filters.text & ~filters.private & ~filters.create(group_only_guard))
async def blocked_group(_, m: Message):
    # other groups: ignore or show message
    if m.text and not m.text.startswith("/"):
        await m.reply_text("❌ This bot works only in the authorized group.")

@bot.on_callback_query()
async def callbacks(_, cq):
    data = (cq.data or "")
    u = cq.from_user

    try:
        if data.startswith("det|"):
            # Detail -> send to PM
            _, media_type, tid = data.split("|")
            tid = int(tid)
            det = tmdb_details(media_type, tid)

            title = det.get("title") if media_type == "movie" else det.get("name")
            is_avail = available_in_old_db(title or "")

            cap = build_detail_caption(det, media_type, is_avail)
            kb = build_detail_keyboard(media_type, tid, is_avail)

            # poster
            poster_path = det.get("poster_path")
            photo = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

            # Send PM
            try:
                if photo:
                    await bot.send_photo(chat_id=u.id, photo=photo, caption=cap, reply_markup=kb, parse_mode="markdown")
                else:
                    await bot.send_message(chat_id=u.id, text=cap, reply_markup=kb, parse_mode="markdown")
                await cq.answer("Sent details to your PM ✅", show_alert=False)
            except Exception:
                await cq.answer("Open bot in PM first (press Start) ❗", show_alert=True)

            return

        if data.startswith("req|"):
            # User requests (PM button click)
            _, media_type, tid = data.split("|")
            tid = int(tid)

            det = tmdb_details(media_type, tid)
            title = det.get("title") if media_type == "movie" else det.get("name")
            date = det.get("release_date") if media_type == "movie" else det.get("first_air_date")
            year = date[:4] if date else None

            # If now available, block request
            if available_in_old_db(title or ""):
                await cq.answer("Already available ✅", show_alert=True)
                return

            # Check limit
            pending = get_user_requests(u.id)
            if len(pending) >= MAX_REQUESTS_PER_USER:
                kb = build_replace_keyboard(pending)
                await bot.send_message(
                    chat_id=u.id,
                    text=f"ඔයාගේ requests {MAX_REQUESTS_PER_USER}ක් පිරීලා.\nRemove එකක් කරලා අලුත් එක දාන්න:\n\nYour requests are full. Remove one:",
                    reply_markup=kb
                )
                await cq.answer("Requests full. Remove one first.", show_alert=True)
                return

            # Add request
            add_request(u.id, media_type, tid, title or "Unknown", year)

            # Notify admin channel
            msg = (
                f"📥 NEW REQUEST\n\n"
                f"User: {u.mention} (`{u.id}`)\n"
                f"Type: {media_type}\n"
                f"Title: {title} ({year or '----'})\n"
                f"TMDB: https://www.themoviedb.org/{'movie' if media_type=='movie' else 'tv'}/{tid}"
            )
            await bot.send_message(
                chat_id=ADMIN_REQ_CHANNEL_ID,
                text=msg,
                reply_markup=admin_request_keyboard(media_type, tid, u.id),
                disable_web_page_preview=True
            )

            await cq.answer("Request sent ✅", show_alert=True)
            return

        if data.startswith("rmreq|"):
            # remove one pending request
            _, rid = data.split("|", 1)
            from bson import ObjectId
            oid = ObjectId(rid)
            requests_col.update_one({"_id": oid, "user_id": u.id, "status": "pending"}, {"$set": {"status": "cancelled"}})
            await cq.answer("Removed ✅", show_alert=True)
            return

        if data.startswith("adm_done|"):
            # Admin marks done (updates pending -> done)
            _, media_type, tid = data.split("|")
            tid = int(tid)
            mark_done_by_tmdb(media_type, tid)
            await cq.answer("Marked done ✅", show_alert=True)
            return

        if data.startswith("adm_cancel|"):
            # Admin cancels all pending for that item/user (optional)
            parts = data.split("|")
            media_type = parts[1]
            tid = int(parts[2])
            user_id = int(parts[3])
            requests_col.update_many(
                {"user_id": user_id, "media_type": media_type, "tmdb_id": tid, "status": "pending"},
                {"$set": {"status": "cancelled"}}
            )
            await cq.answer("Cancelled ✅", show_alert=True)
            return

    except Exception:
        await cq.answer("Error", show_alert=True)

# -------------------------
# Watcher: when new file arrives in old DB, notify requesters
# -------------------------
def notify_requesters_for_title(title_guess: str):
    # naive: match pending requests by normalized title includes
    norm_new = normalize_title(title_guess)
    if not norm_new:
        return

    # take first 3 words for matching
    key = " ".join(norm_new.split()[:3]).strip()
    if not key:
        return

    # scan pending requests (small volume expected)
    for r in requests_col.find({"status": "pending"}):
        t = r.get("title", "")
        if not t:
            continue
        if key in normalize_title(t):
            user_id = r["user_id"]
            media_type = r["media_type"]
            tmdb_id = r["tmdb_id"]

            # mark done
            requests_col.update_one({"_id": r["_id"]}, {"$set": {"status": "done", "done_at": datetime.now(timezone.utc)}})

            # send PM notify
            try:
                det = tmdb_details(media_type, tmdb_id)
                cap = build_detail_caption(det, media_type, True)
                poster_path = det.get("poster_path")
                photo = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

                text = (
                    "✅ Now available in our bot!\n"
                    "✅ දැන් අපේ බොට් එකේ තියෙනවා!\n\n"
                    "Please search in the main movie bot.\n"
                    "Main bot එකෙන් search කරලා ගන්න."
                )

                # send both: notify + card
                bot.send_message(user_id, text)
                if photo:
                    bot.send_photo(user_id, photo=photo, caption=cap, parse_mode="markdown")
                else:
                    bot.send_message(user_id, cap, parse_mode="markdown")
            except Exception:
                pass

def change_stream_worker():
    while True:
        try:
            # watch inserts
            with files_col.watch([{"$match": {"operationType": "insert"}}], full_document="default") as stream:
                for change in stream:
                    doc = change.get("fullDocument") or {}
                    fn = doc.get("file_name") or ""
                    if fn:
                        notify_requesters_for_title(fn)
        except Exception:
            # reconnect after short sleep
            time.sleep(5)

# -------------------------
# Main
# -------------------------
def main():
    # start Flask
    t = threading.Thread(target=run_web, daemon=True)
    t.start()

    # start change stream watcher
    w = threading.Thread(target=change_stream_worker, daemon=True)
    w.start()

    # run bot
    bot.run()

if __name__ == "__main__":
    main()
