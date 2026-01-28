import os
import re
import threading
from datetime import datetime, timezone

import requests
from flask import Flask
from pymongo import MongoClient
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from bson import ObjectId

# =====================================================
# ENV
# =====================================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")

ALLOWED_GROUP_ID = int(os.getenv("ALLOWED_GROUP_ID"))
ADMIN_REQ_CHANNEL_ID = int(os.getenv("ADMIN_REQ_CHANNEL_ID"))

MONGO_URI = os.getenv("MONGO_URI")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

PORT = int(os.getenv("PORT", "10000"))

# =====================================================
# CONSTANTS
# =====================================================
FILES_DB_NAME = "autofilter"
FILES_COLLECTION = "royal_files"
NEW_DB_NAME = "requestbot"

RESULT_BUTTONS = 10
MAX_REQUESTS = 3

# =====================================================
# Flask (Render port)
# =====================================================
app = Flask(__name__)

@app.route("/")
def home():
    return "BOT IS RUNNING"

def run_flask():
    app.run("0.0.0.0", PORT)

# =====================================================
# MongoDB
# =====================================================
mongo = MongoClient(MONGO_URI)

files_col = mongo[FILES_DB_NAME][FILES_COLLECTION]
new_db = mongo[NEW_DB_NAME]
users_col = new_db["users"]
requests_col = new_db["requests"]

# =====================================================
# Normalization helpers
# =====================================================
BAD_WORDS = {
    "1080p","720p","480p","2160p","4k",
    "webrip","webdl","web-dl","bluray","brrip","hdrip","dvdrip",
    "x264","x265","h264","h265","hevc","aac","ddp","dd","dts",
    "10bit","8bit","hdr","sdr","proper","repack",
    "mkv","mp4","avi"
}

YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")

def normalize_title(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[\[\]\(\)\{\}\|_]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    parts = [p for p in text.split() if p and p not in BAD_WORDS]
    return " ".join(parts).strip()

def extract_year(text: str):
    m = YEAR_RE.search(text or "")
    return m.group(1) if m else None

# =====================================================
# Exact file matcher
# =====================================================
def find_exact_files(title: str, year: str | None, limit: int = 5):
    norm_title = normalize_title(title)
    if not norm_title:
        return []

    first = norm_title.split()[0]
    query = {"file_name": {"$regex": re.escape(first), "$options": "i"}}

    results = []
    for doc in files_col.find(query, {"file_name": 1}).limit(400):
        fn = doc.get("file_name", "")
        fn_norm = normalize_title(fn)

        if fn_norm != norm_title:
            continue

        if year:
            y = extract_year(fn)
            if y != year:
                continue

        results.append(doc)
        if len(results) >= limit:
            break

    return results

# =====================================================
# TMDB
# =====================================================
TMDB = "https://api.themoviedb.org/3"

def tmdb_search(q):
    r = requests.get(
        f"{TMDB}/search/multi",
        params={"api_key": TMDB_API_KEY, "query": q},
        timeout=15
    )
    return [
        x for x in r.json().get("results", [])
        if x.get("media_type") in ("movie", "tv")
    ]

def tmdb_detail(t, i):
    r = requests.get(
        f"{TMDB}/{t}/{i}",
        params={"api_key": TMDB_API_KEY},
        timeout=15
    )
    return r.json()

# =====================================================
# Telegram Bot
# =====================================================
bot = Client(
    "request_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# =====================================================
# Handlers
# =====================================================
@bot.on_message(filters.command("start"))
async def start(_, m):
    users_col.update_one(
        {"_id": m.from_user.id},
        {"$set": {
            "name": m.from_user.first_name,
            "username": m.from_user.username,
            "last_seen": datetime.now(timezone.utc)
        }},
        upsert=True
    )
    await m.reply(
        "👋 Welcome!\n\n"
        "🔍 Search movies / series in the main group.\n"
        "📩 Details will arrive here.\n\n"
        "Group එකෙන් search කරන්න."
    )

@bot.on_message(filters.text & filters.group)
async def group_search(_, m):
    if m.chat.id != ALLOWED_GROUP_ID:
        return

    q = m.text.strip()
    if not q or q.startswith("/"):
        return

    results = tmdb_search(q)
    if not results:
        await m.reply("❌ No results found.")
        return

    buttons = []
    for r in results[:RESULT_BUTTONS]:
        title = r.get("title") or r.get("name")
        date = r.get("release_date") or r.get("first_air_date") or ""
        year = date[:4] if date else "----"
        buttons.append([
            InlineKeyboardButton(
                f"{title} ({year})",
                callback_data=f"det|{r['media_type']}|{r['id']}"
            )
        ])

    await m.reply(
        "🎬 Select one:",
        reply_markup=InlineKeyboardMarkup(buttons)
    )

@bot.on_callback_query()
async def callbacks(_, cq):
    data = cq.data
    uid = cq.from_user.id

    # ---------------- Details ----------------
    if data.startswith("det|"):
        _, t, tid = data.split("|")
        det = tmdb_detail(t, tid)

        title = det.get("title") or det.get("name")
        date = det.get("release_date") or det.get("first_air_date") or ""
        year = date[:4] if date else None

        files = find_exact_files(title, year, 5)
        available = len(files) > 0

        if available:
            file_lines = "\n".join([f"• `{f['file_name']}`" for f in files])
            msg = (
                f"🎬 **{title} ({year})**\n\n"
                f"✅ Available in our bot / තියෙනවා\n\n"
                f"📁 Files:\n{file_lines}"
            )
            buttons = None
        else:
            msg = (
                f"🎬 **{title} ({year})**\n\n"
                f"❌ Not available / නැහැ"
            )
            buttons = InlineKeyboardMarkup([
                [InlineKeyboardButton("📥 Request", callback_data=f"req|{t}|{tid}")]
            ])

        await bot.send_message(uid, msg, reply_markup=buttons)
        await cq.answer("Sent to PM")

    # ---------------- Request ----------------
    elif data.startswith("req|"):
        _, t, tid = data.split("|")
        det = tmdb_detail(t, tid)
        title = det.get("title") or det.get("name")
        date = det.get("release_date") or det.get("first_air_date") or ""
        year = date[:4] if date else None

        if len(find_exact_files(title, year, 1)) > 0:
            await cq.answer("Already available ✅", show_alert=True)
            return

        count = requests_col.count_documents({"user": uid, "status": "pending"})
        if count >= MAX_REQUESTS:
            old = list(requests_col.find(
                {"user": uid, "status": "pending"}
            ).limit(MAX_REQUESTS))

            rows = []
            for r in old:
                rows.append([
                    InlineKeyboardButton(
                        f"🗑 Remove {r['title']}",
                        callback_data=f"rm|{r['_id']}"
                    )
                ])

            await bot.send_message(
                uid,
                "⚠️ Request limit full (3).\nRemove one to add new:",
                reply_markup=InlineKeyboardMarkup(rows)
            )
            await cq.answer("Limit reached", show_alert=True)
            return

        requests_col.insert_one({
            "user": uid,
            "title": title,
            "year": year,
            "tmdb_id": tid,
            "type": t,
            "status": "pending",
            "time": datetime.now(timezone.utc)
        })

        await bot.send_message(
            ADMIN_REQ_CHANNEL_ID,
            f"📥 NEW REQUEST\n\nUser: {uid}\nTitle: {title} ({year})"
        )

        await cq.answer("Request sent ✅", show_alert=True)

    # ---------------- Remove request ----------------
    elif data.startswith("rm|"):
        _, rid = data.split("|", 1)
        requests_col.update_one(
            {"_id": ObjectId(rid), "user": uid},
            {"$set": {"status": "cancelled"}}
        )
        await cq.answer("Removed ✅", show_alert=True)

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    bot.run()
