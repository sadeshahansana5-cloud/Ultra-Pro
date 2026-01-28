import os
import re
import math
import json
import time
import threading
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple

import requests
from flask import Flask
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pyrogram import Client, filters
from pyrogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
)

# ============================================================
# ENV + SETTINGS
# ============================================================
def env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default

def env_str(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip()

BOT_TOKEN = env_str("BOT_TOKEN")
API_ID = env_int("API_ID")
API_HASH = env_str("API_HASH")

ALLOWED_GROUP_ID = env_int("ALLOWED_GROUP_ID")
ADMIN_REQ_CHANNEL_ID = env_int("ADMIN_REQ_CHANNEL_ID")

MONGO_URI = env_str("MONGO_URI")
TMDB_API_KEY = env_str("TMDB_API_KEY")

PORT = env_int("PORT", 10000)

FILES_DB_NAME = env_str("FILES_DB_NAME", "autofilter")
FILES_COLLECTION = env_str("FILES_COLLECTION", "royal_files")
NEW_DB_NAME = env_str("NEW_DB_NAME", "requestbot")

RESULT_BUTTONS = env_int("RESULT_BUTTONS", 10)
MAX_REQUESTS = env_int("MAX_REQUESTS", env_int("MAX_REQUESTS_PER_USER", 3))

# Safe minimums
if RESULT_BUTTONS <= 0:
    RESULT_BUTTONS = 10
if MAX_REQUESTS <= 0:
    MAX_REQUESTS = 3

# ============================================================
# Flask (Render port binding)
# ============================================================
web_app = Flask(__name__)

@web_app.get("/")
def home():
    return {"ok": True, "service": "tmdb-request-bot"}

def run_web():
    web_app.run(host="0.0.0.0", port=PORT)

# ============================================================
# Mongo
# ============================================================
mongo = None
files_col = None
users_col = None
requests_col = None

def init_mongo():
    global mongo, files_col, users_col, requests_col
    mongo = MongoClient(MONGO_URI, connectTimeoutMS=10000, serverSelectionTimeoutMS=10000)

    files_col = mongo[FILES_DB_NAME][FILES_COLLECTION]
    new_db = mongo[NEW_DB_NAME]
    users_col = new_db["users"]
    requests_col = new_db["requests"]

# ============================================================
# Text cleaning / Fuzzy match for messy file_name
# ============================================================
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
SEASON_EP_RE = re.compile(r"\bS(\d{1,2})E(\d{1,2})\b", re.IGNORECASE)
SEASON_ONLY_RE = re.compile(r"\bS(\d{1,2})\b", re.IGNORECASE)

# Common junk tokens found in file names
JUNK_WORDS = {
    # quality / codec
    "1080p","720p","480p","2160p","4k","hdr","sdr","10bit","8bit",
    "webrip","webdl","web-dl","web","hdrip","brrip","bluray","dvdrip","cam","ts","telesync",
    "x264","x265","h264","h265","hevc","av1",
    "aac","dd","ddp","dts","eac3","ac3","atmos",
    "mkv","mp4","avi","mov",
    # release tags
    "proper","repack","extended","uncut","remux",
    # language-ish tags (we don't want these to affect title match)
    "tamil","telugu","hindi","malayalam","kannada","english","sinhala","dubbed","dubbing","sub","subs","subtitle","subtitles",
    "hq","hd","sd",
    # random common channel words
    "channel","movie","movies","film","films","series","season","ep","episode",
    # domains / noise
    "www","com","net","org","lk","io","app","to","me","co","tv",
}

# Extra patterns that often appear as prefixes
PREFIX_JUNK_PATTERNS = [
    r"^@[\w\d_]+",            # @CC
    r"^[\w\d_]{1,8}\s",       # A2M , A2M The Avengers...
    r"^cine\w+\s",            # CineSubz...
    r"^mov\w+\s",             # MovCr...
]

def strip_prefix_junk(s: str) -> str:
    s2 = s.strip()
    for pat in PREFIX_JUNK_PATTERNS:
        s2 = re.sub(pat, "", s2, flags=re.IGNORECASE).strip()
    return s2

def normalize_for_match(text: str) -> Tuple[str, List[str], Optional[str]]:
    """
    Returns:
      normalized string (space-joined tokens),
      tokens list,
      detected year (if any)
    """
    t = (text or "").strip()
    t = strip_prefix_junk(t)

    t = t.lower()

    # Replace separators
    t = re.sub(r"[\[\]\(\)\{\}\|_]", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)  # keep letters/numbers/underscore -> then split

    # Remove multi-space
    t = re.sub(r"\s+", " ", t).strip()

    year = None
    m = YEAR_RE.search(t)
    if m:
        year = m.group(1)

    raw_tokens = [x for x in t.split(" ") if x]
    tokens = []
    for tok in raw_tokens:
        if tok in JUNK_WORDS:
            continue
        # drop tokens that are just digits but not year
        if tok.isdigit() and (len(tok) != 4):
            continue
        # drop very short noise
        if len(tok) <= 1:
            continue
        tokens.append(tok)

    norm = " ".join(tokens).strip()
    return norm, tokens, year

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0

def seq_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def score_title_match(tmdb_title: str, file_name: str, tmdb_year: Optional[str]) -> Tuple[float, bool]:
    """
    Returns (score, year_ok)
    score in [0..1]
    """
    t_norm, t_tokens, _ = normalize_for_match(tmdb_title)
    f_norm, f_tokens, f_year = normalize_for_match(file_name)

    if not t_norm or not f_norm:
        return 0.0, False

    t_set = set(t_tokens)
    f_set = set(f_tokens)

    jac = jaccard(t_set, f_set)
    seq = seq_ratio(t_norm, f_norm)

    # Year rule:
    # - If tmdb_year exists: require exact year match in filename (strict)
    year_ok = True
    if tmdb_year:
        year_ok = (f_year == tmdb_year)

    # Combined score with heavier weight on token overlap
    score = 0.65 * jac + 0.35 * seq

    # If year mismatch, heavily penalize
    if tmdb_year and not year_ok:
        score *= 0.15

    # If tokens overlap strongly, boost slightly
    if jac >= 0.85:
        score = min(1.0, score + 0.08)

    return score, year_ok

def find_best_files_for_title(
    title: str,
    year: Optional[str],
    limit: int = 5,
    scan_limit: int = 1200
) -> List[Dict[str, Any]]:
    """
    Fuzzy title match with STRICT year match if year is given.
    Returns top matches sorted by score.
    """
    if not files_col:
        return []

    # Prefilter by first meaningful token
    t_norm, t_tokens, _ = normalize_for_match(title)
    if not t_tokens:
        return []

    first = t_tokens[0]
    # regex prefilter helps avoid scanning 20k fully
    q = {"file_name": {"$regex": re.escape(first), "$options": "i"}}

    candidates = []
    try:
        cursor = files_col.find(q, {"file_name": 1}).limit(scan_limit)
        for doc in cursor:
            fn = doc.get("file_name", "") or ""
            sc, year_ok = score_title_match(title, fn, year)
            # Threshold:
            # - if year exists: require score >= 0.55 (and year_ok already folded, but keep threshold)
            # - if no year: require score >= 0.70
            if year:
                if sc >= 0.55:
                    candidates.append({"file_name": fn, "score": sc})
            else:
                if sc >= 0.70:
                    candidates.append({"file_name": fn, "score": sc})
    except PyMongoError:
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    # Return top unique by file_name
    seen = set()
    out = []
    for c in candidates:
        if c["file_name"] in seen:
            continue
        seen.add(c["file_name"])
        out.append(c)
        if len(out) >= limit:
            break
    return out

# ============================================================
# TMDB (rich details)
# ============================================================
TMDB_BASE = "https://api.themoviedb.org/3"

def tmdb_get(path: str, params: Optional[dict] = None) -> dict:
    params = params or {}
    params["api_key"] = TMDB_API_KEY
    params.setdefault("language", "en-US")
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def tmdb_search_multi(query: str) -> List[dict]:
    data = tmdb_get("/search/multi", {"query": query, "include_adult": "false", "page": 1})
    items = []
    for it in data.get("results", []):
        mt = it.get("media_type")
        if mt in ("movie", "tv"):
            items.append(it)
    return items

def tmdb_fetch_full(media_type: str, tmdb_id: int) -> dict:
    # append extra info
    append = "credits,external_ids,images,content_ratings,release_dates"
    return tmdb_get(f"/{media_type}/{tmdb_id}", {"append_to_response": append})

def tmdb_poster_url(poster_path: Optional[str]) -> Optional[str]:
    if not poster_path:
        return None
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

def pick_director(credits: dict) -> Optional[str]:
    crew = credits.get("crew", []) if credits else []
    for c in crew:
        if (c.get("job") or "").lower() == "director":
            return c.get("name")
    return None

def top_cast(credits: dict, n: int = 8) -> List[str]:
    cast = credits.get("cast", []) if credits else []
    names = []
    for c in cast[:n]:
        nm = c.get("name")
        if nm:
            names.append(nm)
    return names

def format_runtime(minutes: Optional[int]) -> str:
    if not minutes:
        return "-"
    h = minutes // 60
    m = minutes % 60
    if h <= 0:
        return f"{m} min"
    return f"{h}h {m}m"

def safe(s: Any, default: str = "-") -> str:
    if s is None:
        return default
    if isinstance(s, str) and not s.strip():
        return default
    return str(s)

# ============================================================
# Requests system
# ============================================================
def now_utc():
    return datetime.now(timezone.utc)

def pending_count(user_id: int) -> int:
    try:
        return requests_col.count_documents({"user_id": user_id, "status": "pending"})
    except Exception:
        return 0

def get_pending(user_id: int) -> List[dict]:
    try:
        return list(requests_col.find({"user_id": user_id, "status": "pending"}).sort("created_at", -1).limit(MAX_REQUESTS))
    except Exception:
        return []

def insert_request(user_id: int, media_type: str, tmdb_id: int, title: str, year: Optional[str]):
    requests_col.insert_one({
        "user_id": user_id,
        "media_type": media_type,
        "tmdb_id": tmdb_id,
        "title": title,
        "year": year,
        "status": "pending",
        "created_at": now_utc()
    })

def cancel_request(user_id: int, req_id: str):
    requests_col.update_one(
        {"_id": ObjectId(req_id), "user_id": user_id, "status": "pending"},
        {"$set": {"status": "cancelled", "cancelled_at": now_utc()}}
    )

# ============================================================
# Telegram bot
# ============================================================
bot = Client(
    "ultra_tmdb_request_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

BOT_USERNAME_CACHE = {"username": None}

async def get_bot_username() -> str:
    if BOT_USERNAME_CACHE["username"]:
        return BOT_USERNAME_CACHE["username"]
    me = await bot.get_me()
    BOT_USERNAME_CACHE["username"] = me.username
    return me.username

def start_link(username: str) -> str:
    return f"https://t.me/{username}?start=go"

def in_allowed_group(m: Message) -> bool:
    return bool(m.chat and m.chat.type in ("group", "supergroup") and m.chat.id == ALLOWED_GROUP_ID)

# ============================================================
# UI builders
# ============================================================
def kb_start_pm(username: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔓 Start bot in PM (Required)", url=start_link(username))]
    ])

def build_search_keyboard(items: List[dict]) -> InlineKeyboardMarkup:
    rows = []
    for it in items[:RESULT_BUTTONS]:
        mt = it.get("media_type")
        tid = it.get("id")
        title = it.get("title") if mt == "movie" else it.get("name")
        date = it.get("release_date") if mt == "movie" else it.get("first_air_date")
        year = date[:4] if date else "----"
        icon = "🎬" if mt == "movie" else "📺"
        rows.append([InlineKeyboardButton(f"{icon} {title} ({year})", callback_data=f"det|{mt}|{tid}")])
    return InlineKeyboardMarkup(rows)

def build_replace_keyboard(reqs: List[dict]) -> InlineKeyboardMarkup:
    rows = []
    for r in reqs[:MAX_REQUESTS]:
        rid = str(r["_id"])
        title = r.get("title", "Unknown")
        year = r.get("year") or ""
        rows.append([InlineKeyboardButton(f"🗑 Remove: {title} {year}".strip(), callback_data=f"rm|{rid}")])
    return InlineKeyboardMarkup(rows)

def build_request_button(media_type: str, tmdb_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📥 Request this (ඉල්ලන්න)", callback_data=f"req|{media_type}|{tmdb_id}")]
    ])

# ============================================================
# Card builder (Rich TMDB info)
# ============================================================
def build_detail_text(full: dict, media_type: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (text, title, year)
    """
    if media_type == "movie":
        title = full.get("title") or "Unknown"
        date = full.get("release_date") or ""
        runtime = full.get("runtime")
        season_line = ""
        runtime_line = f"⏱ Runtime: **{format_runtime(runtime)}**"
    else:
        title = full.get("name") or "Unknown"
        date = full.get("first_air_date") or ""
        seasons = full.get("number_of_seasons")
        episodes = full.get("number_of_episodes")
        runtime_line = ""
        season_line = f"📺 Seasons: **{safe(seasons)}** | Episodes: **{safe(episodes)}**"

    year = date[:4] if date else None
    rating = safe(full.get("vote_average"))
    votes = safe(full.get("vote_count"))
    status = safe(full.get("status"))
    genres = ", ".join([g["name"] for g in full.get("genres", [])]) if full.get("genres") else "-"
    overview = full.get("overview") or "No overview available."

    credits = full.get("credits", {}) or {}
    director = pick_director(credits) if media_type == "movie" else None
    cast_list = top_cast(credits, 8)
    cast = ", ".join(cast_list) if cast_list else "-"

    ext = full.get("external_ids", {}) or {}
    imdb_id = ext.get("imdb_id")
    imdb_line = f"🎭 IMDb ID: `{imdb_id}`" if imdb_id else "🎭 IMDb ID: -"

    homepage = full.get("homepage") or ""
    tmdb_id = full.get("id")
    tmdb_link = f"https://www.themoviedb.org/{'movie' if media_type=='movie' else 'tv'}/{tmdb_id}"

    # Some extra fields
    orig_lang = safe(full.get("original_language"))
    countries = "-"
    if media_type == "movie":
        pcs = full.get("production_countries") or []
        if pcs:
            countries = ", ".join([c.get("name") for c in pcs if c.get("name")]) or "-"
    else:
        pcs = full.get("origin_country") or []
        if pcs:
            countries = ", ".join(pcs) or "-"

    extra_lines = []
    if runtime_line:
        extra_lines.append(runtime_line)
    if season_line:
        extra_lines.append(season_line)
    if director:
        extra_lines.append(f"🎬 Director: **{director}**")

    extra_block = "\n".join(extra_lines).strip()

    text = (
        f"**{title}** {f'({year})' if year else ''}\n"
        f"⭐ Rating: **{rating}** ({votes} votes)\n"
        f"🎭 Genres: {genres}\n"
        f"🌍 Country: {countries}\n"
        f"🗣 Language: `{orig_lang}`\n"
        f"📌 Status: **{status}**\n"
        f"{extra_block}\n\n"
        f"👥 Cast: {cast}\n\n"
        f"📝 Overview:\n{overview}\n\n"
        f"{imdb_line}\n"
        f"🔗 TMDB: {tmdb_link}\n"
        f"{('🏠 Homepage: ' + homepage) if homepage else ''}"
    ).strip()

    poster = tmdb_poster_url(full.get("poster_path"))
    return text, title, year, poster, tmdb_link

# ============================================================
# Handlers
# ============================================================
@bot.on_message(filters.command("start"))
async def cmd_start(_, m: Message):
    # Save user
    try:
        users_col.update_one(
            {"_id": m.from_user.id},
            {"$set": {
                "name": m.from_user.first_name,
                "username": m.from_user.username,
                "last_seen": now_utc()
            }, "$setOnInsert": {"first_seen": now_utc()}},
            upsert=True
        )
    except Exception:
        pass

    text = (
        "👋 **Welcome to Ultra Movie Finder Bot!**\n\n"
        "✅ **How it works:**\n"
        "1) Go to the main group and type a movie/series name.\n"
        "2) You will get up to **10 results** as buttons.\n"
        "3) Tap a result → I will send a **full details card here in PM**.\n"
        "4) I will also check if the exact title exists in our database.\n"
        "   - If available ✅ you will see up to **5 matching files**.\n"
        "   - If not available ❌ you can request it.\n\n"
        "📌 **Important:**\n"
        "• Search works **only in the authorized group**.\n"
        "• Details arrive **here (PM)**.\n"
        "• If you can’t receive PM cards, press /start again.\n\n"
        "—\n"
        "👋 **සාදරයෙන් පිළිගනිමු!**\n"
        "✅ **භාවිතා කරන විදිහ:**\n"
        "1) Group එකට ගිහිං movie/series නමක් type කරන්න.\n"
        "2) Results buttons 10ක් එයි.\n"
        "3) එකක් තෝරලා click කලොත් detail card එක මෙහි PM එකට එයි.\n"
        "4) DB එකේ තියෙනවද කියලා smart matching එකෙන් බලනවා.\n"
        "   - තියෙනවා නම් ✅ files 5ක් තෙක් පෙන්වනවා\n"
        "   - නැත්තම් ❌ request කරන්න පුළුවන්\n"
    )
    await m.reply_text(text, disable_web_page_preview=True)

@bot.on_message(filters.command("id") & filters.group)
async def cmd_id(_, m: Message):
    await m.reply_text(f"✅ Chat ID: `{m.chat.id}`", quote=True)

@bot.on_message(filters.text & filters.group)
async def group_search(_, m: Message):
    # Only allowed group
    if not in_allowed_group(m):
        return

    q = (m.text or "").strip()
    if not q or q.startswith("/"):
        return

    # TMDB search
    try:
        items = tmdb_search_multi(q)
    except Exception:
        await m.reply_text("❌ TMDB error. Try again later.")
        return

    if not items:
        await m.reply_text("❌ No results found.")
        return

    kb = build_search_keyboard(items)
    await m.reply_text("🎬 Select the correct title:", reply_markup=kb)

@bot.on_callback_query()
async def cb_handler(_, cq: CallbackQuery):
    data = cq.data or ""
    user_id = cq.from_user.id

    # Ensure username for PM start link
    username = await get_bot_username()

    # ---------------------------
    # DETAILS
    # ---------------------------
    if data.startswith("det|"):
        try:
            _, media_type, tid_s = data.split("|")
            tmdb_id = int(tid_s)
        except Exception:
            await cq.answer("Bad data", show_alert=True)
            return

        # TMDB full
        try:
            full = tmdb_fetch_full(media_type, tmdb_id)
        except Exception:
            await cq.answer("TMDB error", show_alert=True)
            return

        text, title, year, poster, tmdb_link = build_detail_text(full, media_type)

        # Fuzzy DB match (year-aware strict)
        files = []
        try:
            files = find_best_files_for_title(title, year, limit=5)
        except Exception:
            files = []

        available = len(files) > 0
        if available:
            avail_line = "✅ Available in our bot / අපේ බොට් එකේ තියෙනවා ✅"
            file_lines = "\n".join([f"• `{f['file_name']}`  _(score {f['score']:.2f})_" for f in files])
            db_block = f"\n\n{avail_line}\n\n📁 **Matching files (up to 5):**\n{file_lines}"
            reply_kb = None
        else:
            avail_line = "❌ Not available / අපේ බොට් එකේ නැහැ ❌"
            db_block = f"\n\n{avail_line}\n\n💡 You can request this title below.\n💡 පහළ බටන් එකෙන් request කරන්න පුළුවන්."
            reply_kb = build_request_button(media_type, tmdb_id)

        final_text = (text + db_block).strip()

        # Send to PM (if user didn't start bot, this will fail)
        try:
            if poster:
                await bot.send_photo(user_id, poster, caption=final_text, parse_mode="markdown", reply_markup=reply_kb)
            else:
                await bot.send_message(user_id, final_text, parse_mode="markdown", reply_markup=reply_kb, disable_web_page_preview=True)
            await cq.answer("📩 Sent to PM", show_alert=False)
        except Exception:
            # PM blocked -> tell user to start bot
            await cq.answer("Open bot in PM first!", show_alert=True)
            try:
                await cq.message.reply_text(
                    "⚠️ I can’t send the detail card to your PM.\n\n"
                    "✅ Please press the button below and click **START** in PM, then try again.\n\n"
                    "⚠️ PM එකට card එක යවන්න බෑ.\n"
                    "✅ පහළ button එකෙන් PM open කරලා **START** කරලා ආයෙ try කරන්න.",
                    reply_markup=kb_start_pm(username)
                )
            except Exception:
                pass
        return

    # ---------------------------
    # REQUEST
    # ---------------------------
    if data.startswith("req|"):
        try:
            _, media_type, tid_s = data.split("|")
            tmdb_id = int(tid_s)
        except Exception:
            await cq.answer("Bad data", show_alert=True)
            return

        # Load TMDB again for title/year
        try:
            full = tmdb_fetch_full(media_type, tmdb_id)
        except Exception:
            await cq.answer("TMDB error", show_alert=True)
            return

        # Build basics
        if media_type == "movie":
            title = full.get("title") or "Unknown"
            date = full.get("release_date") or ""
        else:
            title = full.get("name") or "Unknown"
            date = full.get("first_air_date") or ""
        year = date[:4] if date else None

        # If became available, block request
        try:
            now_files = find_best_files_for_title(title, year, limit=1)
            if now_files:
                await cq.answer("Already available ✅ / දැනටම තියෙනවා ✅", show_alert=True)
                return
        except Exception:
            pass

        # Limit logic
        cnt = pending_count(user_id)
        if cnt >= MAX_REQUESTS:
            reqs = get_pending(user_id)
            await cq.answer("Requests full", show_alert=True)
            try:
                await bot.send_message(
                    user_id,
                    f"⚠️ Request limit = {MAX_REQUESTS}\n"
                    f"ඔයාට requests {MAX_REQUESTS}ක් පිරීලා.\n"
                    f"පරණ එකක් remove කරලා අලුත් එක add කරන්න:",
                    reply_markup=build_replace_keyboard(reqs)
                )
            except Exception:
                pass
            return

        # Insert request
        try:
            insert_request(user_id, media_type, tmdb_id, title, year)
        except Exception:
            await cq.answer("DB error", show_alert=True)
            return

        # Notify admin channel
        tmdb_link = f"https://www.themoviedb.org/{'movie' if media_type=='movie' else 'tv'}/{tmdb_id}"
        msg = (
            f"📥 **NEW REQUEST**\n\n"
            f"👤 User: `{user_id}`\n"
            f"🎬 Type: `{media_type}`\n"
            f"📝 Title: **{title}** {f'({year})' if year else ''}\n"
            f"🔗 TMDB: {tmdb_link}\n"
        )

        try:
            await bot.send_message(ADMIN_REQ_CHANNEL_ID, msg, disable_web_page_preview=True)
        except Exception:
            # If channel wrong / bot not admin -> inform user
            await cq.answer("Request saved, but admin channel failed!", show_alert=True)
            try:
                await bot.send_message(
                    user_id,
                    "✅ Request saved.\n"
                    "⚠️ But I couldn't send it to admin channel. (Check ADMIN_REQ_CHANNEL_ID / bot admin rights)\n\n"
                    "✅ Request එක save වුණා.\n"
                    "⚠️ Admin channel එකට යවන්න බැරි උනා. (ID හරිද / bot admin ද?)"
                )
            except Exception:
                pass
            return

        await cq.answer("Request sent ✅ / Request එක යැවුණා ✅", show_alert=True)
        return

    # ---------------------------
    # REMOVE OLD REQUEST (to add new)
    # ---------------------------
    if data.startswith("rm|"):
        _, rid = data.split("|", 1)
        try:
            cancel_request(user_id, rid)
            await cq.answer("Removed ✅ / අයින් කළා ✅", show_alert=True)
        except Exception:
            await cq.answer("Error", show_alert=True)
        return

# ============================================================
# MAIN
# ============================================================
def main():
    # Basic env sanity (don’t crash hard; print + keep web alive)
    problems = []
    if not BOT_TOKEN: problems.append("BOT_TOKEN missing")
    if not API_HASH or API_ID == 0: problems.append("API_ID/API_HASH missing")
    if not MONGO_URI: problems.append("MONGO_URI missing")
    if not TMDB_API_KEY: problems.append("TMDB_API_KEY missing")
    if ALLOWED_GROUP_ID == 0: problems.append("ALLOWED_GROUP_ID missing")
    if ADMIN_REQ_CHANNEL_ID == 0: problems.append("ADMIN_REQ_CHANNEL_ID missing")

    # start web server always so Render sees port
    threading.Thread(target=run_web, daemon=True).start()

    if problems:
        print("CONFIG PROBLEMS:", problems)
        # keep process alive for logs
        while True:
            time.sleep(60)

    # init mongo
    try:
        init_mongo()
        # quick ping
        mongo.admin.command("ping")
        print("MongoDB connected OK")
    except Exception as e:
        print("MongoDB init failed:", repr(e))
        # keep alive (Render port still up)
        while True:
            time.sleep(60)

    # run bot
    bot.run()

if __name__ == "__main__":
    main()
