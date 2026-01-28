import os
import re
import threading
from datetime import datetime, timezone

import requests
from flask import Flask
from pymongo import MongoClient
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# ================= ENV =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")

ALLOWED_GROUP_ID = int(os.getenv("ALLOWED_GROUP_ID"))
ADMIN_REQ_CHANNEL_ID = int(os.getenv("ADMIN_REQ_CHANNEL_ID"))

MONGO_URI = os.getenv("MONGO_URI")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

PORT = int(os.getenv("PORT", "10000"))

# ================= CONSTANTS =================
FILES_DB_NAME = "autofilter"
FILES_COLLECTION = "royal_files"
NEW_DB_NAME = "requestbot"

RESULT_BUTTONS = 10
MAX_REQUESTS = 3

# ================= Flask (Render port) =================
app = Flask(__name__)

@app.route("/")
def home():
    return "OK"

def run_flask():
    app.run("0.0.0.0", PORT)

# ================= Mongo =================
mongo = MongoClient(MONGO_URI)

files_col = mongo[FILES_DB_NAME][FILES_COLLECTION]
new_db = mongo[NEW_DB_NAME]
users_col = new_db["users"]
requests_col = new_db["requests"]

# ================= Utils =================
BAD_WORDS = {
    "1080p","720p","480p","2160p","4k","webrip","webdl","bluray",
    "x264","x265","h264","h265","aac","dts","hdrip","brrip"
}

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\[\]\(\)\.\-_]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(w for w in text.split() if w not in BAD_WORDS)

def exists_in_autofilter(title: str) -> bool:
    key = normalize(title)
    if not key:
        return False
    for f in files_col.find(
        {"file_name": {"$regex": key.split()[0], "$options": "i"}},
        {"file_name": 1}
    ).limit(50):
        if key in normalize(f.get("file_name", "")):
            return True
    return False

# ================= TMDB =================
TMDB = "https://api.themoviedb.org/3"

def tmdb_search(q):
    r = requests.get(
        f"{TMDB}/search/multi",
        params={"api_key": TMDB_API_KEY, "query": q},
        timeout=15
    )
    return [x for x in r.json().get("results", []) if x["media_type"] in ["movie","tv"]]

def tmdb_detail(t, i):
    r = requests.get(
        f"{TMDB}/{t}/{i}",
        params={"api_key": TMDB_API_KEY},
        timeout=15
    )
    return r.json()

# ================= Bot =================
bot = Client(
    "request_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ================= Handlers =================
@bot.on_message(filters.command("start"))
async def start(_, m):
    users_col.update_one(
        {"_id": m.from_user.id},
        {"$set": {
            "name": m.from_user.first_name,
            "last_seen": datetime.now(timezone.utc)
        }},
        upsert=True
    )
    await m.reply(
        "👋 Welcome!\n\n"
        "🔍 Search movies/series in the main group.\n"
        "📩 Details will be sent here.\n\n"
        "මෙම බොට් group එක තුළ search සඳහා පමණි."
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
        await m.reply("No results found.")
        return

    buttons = []
    for r in results[:RESULT_BUTTONS]:
        title = r.get("title") or r.get("name")
        year = (r.get("release_date") or r.get("first_air_date") or "")[:4]
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

    if data.startswith("det|"):
        _, t, tid = data.split("|")
        det = tmdb_detail(t, tid)
        title = det.get("title") or det.get("name")

        available = exists_in_autofilter(title)

        msg = (
            f"🎬 **{title}**\n\n"
            f"{'✅ Available in our bot / තියෙනවා' if available else '❌ Not available / නැහැ'}"
        )

        buttons = []
        if not available:
            buttons.append([
                InlineKeyboardButton("📥 Request", callback_data=f"req|{t}|{tid}")
            ])

        await bot.send_message(
            uid,
            msg,
            reply_markup=InlineKeyboardMarkup(buttons) if buttons else None
        )
        await cq.answer("Sent to PM")

    elif data.startswith("req|"):
        _, t, tid = data.split("|")
        if requests_col.count_documents({"user": uid, "status": "pending"}) >= MAX_REQUESTS:
            await cq.answer("Request limit reached (3)", show_alert=True)
            return

        det = tmdb_detail(t, tid)
        title = det.get("title") or det.get("name")

        requests_col.insert_one({
            "user": uid,
            "title": title,
            "tmdb_id": tid,
            "type": t,
            "status": "pending",
            "time": datetime.now(timezone.utc)
        })

        await bot.send_message(
            ADMIN_REQ_CHANNEL_ID,
            f"📥 NEW REQUEST\nUser: {uid}\nTitle: {title}"
        )

        await cq.answer("Request sent ✅", show_alert=True)

# ================= Main =================
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    bot.run()
