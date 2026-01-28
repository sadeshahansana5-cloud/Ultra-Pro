import os
import re
import time
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId

from pyrogram import Client, filters
from pyrogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Message,
    CallbackQuery
)

# ============================================================
# CONFIG
# ============================================================
def env_str(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip()

def env_int(name: str, default: int = 0) -> int:
    try:
        return int((os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return default

@dataclass
class Config:
    bot_token: str
    api_id: int
    api_hash: str
    allowed_group_id: int
    admin_req_channel_id: int

    mongo_uri: str
    tmdb_api_key: str

    port: int

    files_db_name: str
    files_collection: str
    new_db_name: str

    result_buttons: int
    max_requests: int

    poll_seconds: int
    scan_limit: int

CFG = Config(
    bot_token=env_str("BOT_TOKEN"),
    api_id=env_int("API_ID"),
    api_hash=env_str("API_HASH"),
    allowed_group_id=env_int("ALLOWED_GROUP_ID"),
    admin_req_channel_id=env_int("ADMIN_REQ_CHANNEL_ID"),
    mongo_uri=env_str("MONGO_URI"),
    tmdb_api_key=env_str("TMDB_API_KEY"),
    port=env_int("PORT", 10000),
    files_db_name=env_str("FILES_DB_NAME", "autofilter"),
    files_collection=env_str("FILES_COLLECTION", "royal_files"),
    new_db_name=env_str("NEW_DB_NAME", "requestbot"),
    result_buttons=max(1, env_int("RESULT_BUTTONS", 10)),
    max_requests=max(1, env_int("MAX_REQUESTS", env_int("MAX_REQUESTS_PER_USER", 3))),
    poll_seconds=max(10, env_int("POLL_SECONDS", 25)),
    scan_limit=max(500, env_int("SCAN_LIMIT", 2500)),
)

# ============================================================
# SIMPLE LOGGER
# ============================================================
def log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

# ============================================================
# FLASK (Render port binding)
# ============================================================
web = Flask(__name__)

@web.get("/")
def home():
    return {"ok": True, "service": "ultra-tmdb-autofilter"}

def run_web():
    web.run(host="0.0.0.0", port=CFG.port)

# ============================================================
# MONGO INIT
# ============================================================
mongo: Optional[MongoClient] = None
files_col = None
users_col = None
requests_col = None
meta_col = None

def init_mongo():
    global mongo, files_col, users_col, requests_col, meta_col
    mongo = MongoClient(CFG.mongo_uri, connectTimeoutMS=10000, serverSelectionTimeoutMS=10000)
    files_col = mongo[CFG.files_db_name][CFG.files_collection]

    new_db = mongo[CFG.new_db_name]
    users_col = new_db["users"]
    requests_col = new_db["requests"]
    meta_col = new_db["meta"]

    # No createIndex here (Atlas permission safe)
    # We just rely on small volume for request collections.

def mongo_ping_ok() -> bool:
    try:
        mongo.admin.command("ping")
        return True
    except Exception:
        return False

# ============================================================
# TMDB
# ============================================================
TMDB_BASE = "https://api.themoviedb.org/3"

def tmdb_get(path: str, params: Optional[dict] = None) -> dict:
    params = params or {}
    params["api_key"] = CFG.tmdb_api_key
    params.setdefault("language", "en-US")
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def tmdb_search_multi(query: str) -> List[dict]:
    data = tmdb_get("/search/multi", {"query": query, "include_adult": "false", "page": 1})
    out = []
    for it in data.get("results", []):
        mt = it.get("media_type")
        if mt in ("movie", "tv"):
            out.append(it)
    return out

def tmdb_full(media_type: str, tmdb_id: int) -> dict:
    # rich info similar to imdb bots
    append = "credits,external_ids,images,content_ratings,release_dates"
    return tmdb_get(f"/{media_type}/{tmdb_id}", {"append_to_response": append})

def tmdb_poster(poster_path: Optional[str]) -> Optional[str]:
    if not poster_path:
        return None
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

# ============================================================
# TEXT NORMALIZATION + FUZZY MATCH
# ============================================================
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

JUNK_WORDS = {
    # quality / codec
    "1080p","720p","480p","2160p","4k","hdr","sdr","10bit","8bit",
    "webrip","webdl","web-dl","hdrip","brrip","bluray","dvdrip","cam","ts","telesync",
    "x264","x265","h264","h265","hevc","av1",
    "aac","dd","ddp","dts","eac3","ac3","atmos",
    "proper","repack","extended","uncut","remux",
    "mkv","mp4","avi","mov",

    # language-ish
    "tamil","telugu","hindi","malayalam","kannada","english","sinhala",
    "dubbed","dubbing","sub","subs","subtitle","subtitles","hq","hd","sd",

    # noise / domains
    "www","com","net","org","lk","io","to","me","co","tv",

    # common channel words
    "channel","movie","movies","film","films","series",
}

PREFIX_CLEAN = [
    r"^@[\w\d_]+",            # @CC
    r"^\b[a-z0-9]{1,6}\b\s+", # A2M / PSA / etc at start
    r"^cine\w+\s+",           # cinesubz etc
    r"^mov\w+\s+",            # movcr etc
]

def strip_prefix_noise(s: str) -> str:
    x = (s or "").strip()
    for p in PREFIX_CLEAN:
        x = re.sub(p, "", x, flags=re.IGNORECASE).strip()
    return x

def normalize_tokens(text: str) -> Tuple[str, List[str], Optional[str]]:
    """
    Returns normalized string, tokens list, detected year (if any).
    """
    t = strip_prefix_noise(text).lower()
    t = re.sub(r"[\[\]\(\)\{\}\|_]", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    year = None
    ym = YEAR_RE.search(t)
    if ym:
        year = ym.group(1)

    toks = []
    for tok in t.split():
        if tok in JUNK_WORDS:
            continue
        if tok.isdigit() and len(tok) != 4:
            continue
        if len(tok) <= 1:
            continue
        toks.append(tok)

    norm = " ".join(toks).strip()
    return norm, toks, year

def seq_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return (inter / union) if union else 0.0

def score_match(tmdb_title: str, file_name: str, tmdb_year: Optional[str]) -> float:
    """
    Score in [0..1], with strict year behavior:
    - If tmdb_year exists and file has a different year => heavy penalty.
    - If file has no year and tmdb_year exists => penalty but not zero.
    """
    t_norm, t_toks, _ = normalize_tokens(tmdb_title)
    f_norm, f_toks, f_year = normalize_tokens(file_name)

    if not t_toks or not f_toks:
        return 0.0

    t_set = set(t_toks)
    f_set = set(f_toks)

    jac = jaccard(t_set, f_set)
    seq = seq_ratio(t_norm, f_norm)

    score = 0.68 * jac + 0.32 * seq

    if tmdb_year:
        if f_year is None:
            score *= 0.55
        elif f_year != tmdb_year:
            score *= 0.10  # strict reject-ish
        else:
            score = min(1.0, score + 0.06)

    # Bonus if almost all title tokens included in file tokens
    recall = len(t_set & f_set) / max(1, len(t_set))
    if recall >= 0.90:
        score = min(1.0, score + 0.05)

    return score

def find_best_files(title: str, year: Optional[str], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fuzzy find with prefilter token and score sorting.
    """
    if not files_col:
        return []

    t_norm, t_toks, _ = normalize_tokens(title)
    if not t_toks:
        return []

    first = t_toks[0]
    query = {"file_name": {"$regex": re.escape(first), "$options": "i"}}

    matches: List[Dict[str, Any]] = []
    try:
        cursor = files_col.find(query, {"file_name": 1}).limit(CFG.scan_limit)
        for doc in cursor:
            fn = doc.get("file_name") or ""
            sc = score_match(title, fn, year)

            # Thresholds: require higher if no year
            if year:
                if sc >= 0.62:
                    matches.append({"file_name": fn, "score": sc})
            else:
                if sc >= 0.75:
                    matches.append({"file_name": fn, "score": sc})
    except PyMongoError:
        return []

    matches.sort(key=lambda x: x["score"], reverse=True)

    out = []
    seen = set()
    for m in matches:
        if m["file_name"] in seen:
            continue
        seen.add(m["file_name"])
        out.append(m)
        if len(out) >= limit:
            break
    return out

# ============================================================
# REQUEST SYSTEM
# ============================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def req_pending_count(user_id: int) -> int:
    try:
        return requests_col.count_documents({"user_id": user_id, "status": "pending"})
    except Exception:
        return 0

def req_list_pending(user_id: int) -> List[dict]:
    try:
        return list(requests_col.find({"user_id": user_id, "status": "pending"}).sort("created_at", -1).limit(CFG.max_requests))
    except Exception:
        return []

def req_insert(user_id: int, media_type: str, tmdb_id: int, title: str, year: Optional[str]):
    requests_col.insert_one({
        "user_id": user_id,
        "media_type": media_type,
        "tmdb_id": tmdb_id,
        "title": title,
        "year": year,
        "status": "pending",
        "created_at": now_utc(),
    })

def req_cancel(user_id: int, rid: str):
    requests_col.update_one(
        {"_id": ObjectId(rid), "user_id": user_id, "status": "pending"},
        {"$set": {"status": "cancelled", "cancelled_at": now_utc()}}
    )

def req_mark_done_by_tmdb(media_type: str, tmdb_id: int):
    requests_col.update_many(
        {"media_type": media_type, "tmdb_id": tmdb_id, "status": "pending"},
        {"$set": {"status": "done", "done_at": now_utc()}}
    )

# ============================================================
# TELEGRAM BOT
# ============================================================
bot = Client(
    "ultra_pro_max_bot",
    api_id=CFG.api_id,
    api_hash=CFG.api_hash,
    bot_token=CFG.bot_token
)

BOT_USERNAME: Optional[str] = None

async def get_bot_username() -> str:
    global BOT_USERNAME
    if BOT_USERNAME:
        return BOT_USERNAME
    me = await bot.get_me()
    BOT_USERNAME = me.username
    return BOT_USERNAME

def start_url(username: str) -> str:
    return f"https://t.me/{username}?start=go"

def kb_start_pm(username: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔓 Start bot in PM", url=start_url(username))]])

def kb_search_results(items: List[dict]) -> InlineKeyboardMarkup:
    rows = []
    for it in items[:CFG.result_buttons]:
        mt = it.get("media_type")
        tid = it.get("id")
        title = it.get("title") if mt == "movie" else it.get("name")
        date = it.get("release_date") if mt == "movie" else it.get("first_air_date")
        year = date[:4] if date else "----"
        icon = "🎬" if mt == "movie" else "📺"
        rows.append([InlineKeyboardButton(f"{icon} {title} ({year})", callback_data=f"det|{mt}|{tid}")])
    return InlineKeyboardMarkup(rows)

def kb_request(media_type: str, tmdb_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("📥 Request this / ඉල්ලන්න", callback_data=f"req|{media_type}|{tmdb_id}")]])

def kb_replace(reqs: List[dict]) -> InlineKeyboardMarkup:
    rows = []
    for r in reqs[:CFG.max_requests]:
        rid = str(r["_id"])
        title = r.get("title", "Unknown")
        year = r.get("year") or ""
        rows.append([InlineKeyboardButton(f"🗑 Remove: {title} {year}".strip(), callback_data=f"rm|{rid}")])
    return InlineKeyboardMarkup(rows)

def kb_admin_actions(media_type: str, tmdb_id: int, user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Done", callback_data=f"adm_done|{media_type}|{tmdb_id}")],
        [InlineKeyboardButton("🗑 Cancel for user", callback_data=f"adm_cancel|{media_type}|{tmdb_id}|{user_id}")]
    ])

def is_allowed_group(m: Message) -> bool:
    return bool(m.chat and m.chat.type in ("group", "supergroup") and m.chat.id == CFG.allowed_group_id)

# ============================================================
# RICH CARD BUILDER (IMDb-style)
# ============================================================
def pick_director(credits: dict) -> Optional[str]:
    crew = credits.get("crew", []) if credits else []
    for c in crew:
        if (c.get("job") or "").lower() == "director":
            return c.get("name")
    return None

def top_cast(credits: dict, n: int = 10) -> List[str]:
    cast = credits.get("cast", []) if credits else []
    out = []
    for c in cast[:n]:
        nm = c.get("name")
        if nm:
            out.append(nm)
    return out

def fmt_runtime(mins: Optional[int]) -> str:
    if not mins:
        return "-"
    h = mins // 60
    m = mins % 60
    return f"{h}h {m}m" if h else f"{m}m"

def build_card(full: dict, media_type: str) -> Tuple[str, str, Optional[str], Optional[str], str]:
    """
    Returns: (text, title, year, poster_url, tmdb_link)
    """
    if media_type == "movie":
        title = full.get("title") or "Unknown"
        date = full.get("release_date") or ""
        year = date[:4] if date else None
        runtime = fmt_runtime(full.get("runtime"))
        extra1 = f"⏱ Runtime: **{runtime}**"
    else:
        title = full.get("name") or "Unknown"
        date = full.get("first_air_date") or ""
        year = date[:4] if date else None
        seasons = full.get("number_of_seasons")
        episodes = full.get("number_of_episodes")
        extra1 = f"📺 Seasons: **{seasons}** | Episodes: **{episodes}**"

    rating = full.get("vote_average")
    votes = full.get("vote_count")
    status = full.get("status") or "-"
    genres = ", ".join([g["name"] for g in full.get("genres", [])]) if full.get("genres") else "-"
    overview = full.get("overview") or "No overview available."

    credits = full.get("credits", {}) or {}
    director = pick_director(credits) if media_type == "movie" else None
    cast = ", ".join(top_cast(credits, 10)) or "-"

    ext = full.get("external_ids", {}) or {}
    imdb_id = ext.get("imdb_id") or None

    tmdb_id = full.get("id")
    tmdb_link = f"https://www.themoviedb.org/{'movie' if media_type=='movie' else 'tv'}/{tmdb_id}"
    poster = tmdb_poster(full.get("poster_path"))

    lines = []
    lines.append(f"**{title}** {f'({year})' if year else ''}")
    lines.append(f"⭐ Rating: **{rating}**  |  Votes: `{votes}`")
    lines.append(f"🎭 Genres: {genres}")
    lines.append(f"📌 Status: **{status}**")
    lines.append(extra1)
    if director:
        lines.append(f"🎬 Director: **{director}**")
    lines.append("")
    lines.append(f"👥 Cast: {cast}")
    lines.append("")
    lines.append("📝 **Overview:**")
    lines.append(overview)
    lines.append("")
    if imdb_id:
        lines.append(f"🎭 IMDb ID: `{imdb_id}`")
    lines.append(f"🔗 TMDB: {tmdb_link}")

    return "\n".join(lines), title, year, poster, tmdb_link

# ============================================================
# HANDLERS
# ============================================================
@bot.on_message(filters.command("start"))
async def start_cmd(_, m: Message):
    # store user
    try:
        users_col.update_one(
            {"_id": m.from_user.id},
            {"$set": {
                "name": m.from_user.first_name,
                "username": m.from_user.username,
                "last_seen": now_utc(),
            }, "$setOnInsert": {"first_seen": now_utc()}},
            upsert=True
        )
    except Exception:
        pass

    text = (
        "👋 **Welcome to Ultra Pro Max Movie Finder!**\n\n"
        "✅ **What you can do:**\n"
        "• Group එකේ movie/series නමක් type කරන්න\n"
        "• TMDB results buttons 10ක් එයි\n"
        "• Button එකක් click කලොත් detail card එක **PM** එකට එයි\n"
        "• Bot එක DB එකෙන් smart matching කරලා files 5ක් තෙක් පෙන්වයි\n"
        "• නැත්තම් request කරන්න පුළුවන් (userට max requests 3)\n\n"
        "⚠️ **Important:**\n"
        "• Search works only in the authorized group.\n"
        "• PM cards ලැබෙන්න නම් bot එක මෙහි /start කරලා තියෙන්න ඕන.\n\n"
        "—\n"
        "🟢 **සිංහලෙන්:**\n"
        "• Group එකේ නමක් ගැහුවම buttons එයි\n"
        "• එකක් තෝරලා click කලොත් මේ PM එකට ලොකු විස්තර card එක එයි\n"
        "• DB එකේ තියෙනවද කියලා හොඳම matching එකෙන් බලලා files 5ක් තෙක් දෙනවා\n"
        "• නැත්තම් request button එකෙන් ඉල්ලන්න පුළුවන්\n"
    )
    await m.reply_text(text, disable_web_page_preview=True)

@bot.on_message(filters.command("id") & filters.group)
async def id_cmd(_, m: Message):
    await m.reply_text(f"✅ Chat ID: `{m.chat.id}`", quote=True)

# 🔥 GROUP SEARCH FIXED: listen to all non-private text; then check group id
@bot.on_message(filters.text & ~filters.private)
async def group_search(_, m: Message):
    if not is_allowed_group(m):
        return

    q = (m.text or "").strip()
    if not q or q.startswith("/"):
        return

    try:
        items = tmdb_search_multi(q)
    except Exception as e:
        await m.reply_text("❌ TMDB error. Try again later.")
        return

    if not items:
        await m.reply_text("❌ No results found.")
        return

    await m.reply_text("🎬 Select the correct title:", reply_markup=kb_search_results(items))

@bot.on_callback_query()
async def callbacks(_, cq: CallbackQuery):
    data = cq.data or ""
    uid = cq.from_user.id

    # --------------------- DETAILS ---------------------
    if data.startswith("det|"):
        try:
            _, media_type, tid = data.split("|")
            tmdb_id = int(tid)
        except Exception:
            await cq.answer("Bad request", show_alert=True)
            return

        try:
            full = tmdb_full(media_type, tmdb_id)
        except Exception:
            await cq.answer("TMDB error", show_alert=True)
            return

        card_text, title, year, poster, tmdb_link = build_card(full, media_type)

        # DB matching
        try:
            matches = find_best_files(title, year, limit=5)
        except Exception:
            matches = []

        if matches:
            file_lines = "\n".join([f"• `{m['file_name']}`  _(score {m['score']:.2f})_" for m in matches])
            status_block = (
                "\n\n✅ **Available in our bot / අපේ බොට් එකේ තියෙනවා ✅**\n"
                "📁 **Matching files (up to 5):**\n"
                f"{file_lines}"
            )
            kb = None
        else:
            status_block = (
                "\n\n❌ **Not available / අපේ බොට් එකේ නැහැ ❌**\n"
                "💡 You can request it below.\n"
                "💡 පහළ button එකෙන් request කරන්න."
            )
            kb = kb_request(media_type, tmdb_id)

        final = (card_text + status_block).strip()

        # Send PM (if user didn't start bot, handle gracefully)
        try:
            if poster:
                await bot.send_photo(uid, poster, caption=final, parse_mode="markdown", reply_markup=kb)
            else:
                await bot.send_message(uid, final, parse_mode="markdown", reply_markup=kb, disable_web_page_preview=True)
            await cq.answer("📩 Sent to PM", show_alert=False)
        except Exception:
            await cq.answer("Open bot in PM first!", show_alert=True)
            username = await get_bot_username()
            try:
                await cq.message.reply_text(
                    "⚠️ I can’t send the detail card to your PM.\n"
                    "✅ Please press below button → START in PM → try again.\n\n"
                    "⚠️ PM එකට card යවන්න බැ.\n"
                    "✅ පහළ button එකෙන් PM open කරලා START කරලා ආයෙ try කරන්න.",
                    reply_markup=kb_start_pm(username)
                )
            except Exception:
                pass
        return

    # --------------------- REQUEST ---------------------
    if data.startswith("req|"):
        try:
            _, media_type, tid = data.split("|")
            tmdb_id = int(tid)
        except Exception:
            await cq.answer("Bad request", show_alert=True)
            return

        # limit check
        cnt = req_pending_count(uid)
        if cnt >= CFG.max_requests:
            reqs = req_list_pending(uid)
            await cq.answer("Requests full", show_alert=True)
            try:
                await bot.send_message(
                    uid,
                    f"⚠️ Request limit = {CFG.max_requests}\n"
                    f"ඔයාට requests {CFG.max_requests}ක් පිරීලා.\n"
                    f"පරණ එකක් remove කරලා අලුත් එක add කරන්න:",
                    reply_markup=kb_replace(reqs)
                )
            except Exception:
                pass
            return

        # load TMDB for title/year
        try:
            full = tmdb_full(media_type, tmdb_id)
        except Exception:
            await cq.answer("TMDB error", show_alert=True)
            return

        if media_type == "movie":
            title = full.get("title") or "Unknown"
            date = full.get("release_date") or ""
        else:
            title = full.get("name") or "Unknown"
            date = full.get("first_air_date") or ""
        year = date[:4] if date else None

        # if already available now, block
        try:
            existing = find_best_files(title, year, limit=1)
            if existing:
                await cq.answer("Already available ✅ / දැනටම තියෙනවා ✅", show_alert=True)
                return
        except Exception:
            pass

        # insert
        try:
            req_insert(uid, media_type, tmdb_id, title, year)
        except Exception:
            await cq.answer("DB error", show_alert=True)
            return

        # notify admin channel
        tmdb_link = f"https://www.themoviedb.org/{'movie' if media_type=='movie' else 'tv'}/{tmdb_id}"
        msg = (
            f"📥 **NEW REQUEST**\n\n"
            f"👤 User: `{uid}`\n"
            f"🎬 Type: `{media_type}`\n"
            f"📝 Title: **{title}** {f'({year})' if year else ''}\n"
            f"🔗 TMDB: {tmdb_link}\n"
        )
        try:
            await bot.send_message(CFG.admin_req_channel_id, msg, reply_markup=kb_admin_actions(media_type, tmdb_id, uid), disable_web_page_preview=True)
        except Exception:
            # channel fail -> tell user but keep request saved
            await cq.answer("Saved, but admin channel failed!", show_alert=True)
            try:
                await bot.send_message(
                    uid,
                    "✅ Request saved.\n"
                    "⚠️ Admin channel එකට send වෙන්නේ නැ. (ADMIN_REQ_CHANNEL_ID හරිද / bot admin ද?)\n\n"
                    "✅ Request එක save වුණා.\n"
                    "⚠️ Admin channel දෝෂයක් තියෙනවා."
                )
            except Exception:
                pass
            return

        await cq.answer("Request sent ✅ / Request එක යැවුණා ✅", show_alert=True)
        return

    # --------------------- REMOVE REQUEST ---------------------
    if data.startswith("rm|"):
        _, rid = data.split("|", 1)
        try:
            req_cancel(uid, rid)
            await cq.answer("Removed ✅ / අයින් කළා ✅", show_alert=True)
        except Exception:
            await cq.answer("Error", show_alert=True)
        return

    # --------------------- ADMIN: DONE ---------------------
    if data.startswith("adm_done|"):
        try:
            _, media_type, tid = data.split("|")
            tmdb_id = int(tid)
        except Exception:
            await cq.answer("Bad", show_alert=True)
            return

        try:
            req_mark_done_by_tmdb(media_type, tmdb_id)
        except Exception:
            pass

        await cq.answer("Marked done ✅", show_alert=True)
        return

    # --------------------- ADMIN: CANCEL FOR USER ---------------------
    if data.startswith("adm_cancel|"):
        parts = data.split("|")
        if len(parts) != 4:
            await cq.answer("Bad", show_alert=True)
            return
        media_type = parts[1]
        tmdb_id = int(parts[2])
        user_id = int(parts[3])

        try:
            requests_col.update_many(
                {"user_id": user_id, "media_type": media_type, "tmdb_id": tmdb_id, "status": "pending"},
                {"$set": {"status": "cancelled", "cancelled_at": now_utc()}}
            )
        except Exception:
            pass

        await cq.answer("Cancelled ✅", show_alert=True)
        return

# ============================================================
# AUTO NOTIFY: poll new files from autofilter and match pending requests
# ============================================================
def meta_get_last_oid() -> Optional[ObjectId]:
    try:
        doc = meta_col.find_one({"_id": "last_file_oid"})
        if doc and doc.get("oid"):
            return ObjectId(doc["oid"])
    except Exception:
        return None
    return None

def meta_set_last_oid(oid: ObjectId):
    try:
        meta_col.update_one({"_id": "last_file_oid"}, {"$set": {"oid": str(oid), "updated_at": now_utc()}}, upsert=True)
    except Exception:
        pass

def extract_guess_title_year_from_filename(fn: str) -> Tuple[str, Optional[str]]:
    """
    For matching new files to pending requests:
    - remove junk tokens, keep tokens + year
    - return a 'guess title' string and year
    """
    norm, toks, year = normalize_tokens(fn)
    # keep a reasonable title guess from tokens (up to 8 tokens)
    guess = " ".join(toks[:8]).strip() if toks else ""
    return guess, year

def pending_requests_all(limit: int = 2000) -> List[dict]:
    try:
        return list(requests_col.find({"status": "pending"}).sort("created_at", 1).limit(limit))
    except Exception:
        return []

def notify_user_available(user_id: int, media_type: str, tmdb_id: int):
    """
    Send PM to user with rich card + availability.
    """
    try:
        full = tmdb_full(media_type, tmdb_id)
        card_text, title, year, poster, tmdb_link = build_card(full, media_type)
        matches = find_best_files(title, year, limit=5)
        if matches:
            file_lines = "\n".join([f"• `{m['file_name']}`  _(score {m['score']:.2f})_" for m in matches])
        else:
            file_lines = "• (files detected, but not listed)"

        msg = (
            "✅ **Now available! / දැන් තියෙනවා!**\n\n"
            f"{card_text}\n\n"
            "📁 **Files:**\n"
            f"{file_lines}\n\n"
            "👉 Please search in the main bot/group to get it.\n"
            "👉 Main group එකෙන් search කරලා ගන්න."
        )

        if poster:
            bot.send_photo(user_id, poster, caption=msg, parse_mode="markdown")
        else:
            bot.send_message(user_id, msg, parse_mode="markdown", disable_web_page_preview=True)

    except Exception:
        # user might not have started bot => ignore
        pass

def poll_worker():
    """
    Poll for new inserts in files_col (autofilter royal_files).
    Uses _id(ObjectId) ordering.
    """
    log(f"Auto-notify poller started. Every {CFG.poll_seconds}s")

    last_oid = meta_get_last_oid()

    # If first run and no last oid, set to current latest to avoid spamming old files
    if last_oid is None:
        try:
            latest = files_col.find({}, {"_id": 1}).sort("_id", -1).limit(1)
            latest_doc = next(latest, None)
            if latest_doc:
                last_oid = latest_doc["_id"]
                meta_set_last_oid(last_oid)
                log("Initialized last_oid to latest existing file (no backfill).")
        except Exception:
            pass

    while True:
        try:
            # fetch new docs
            q = {"_id": {"$gt": last_oid}} if last_oid else {}
            cur = files_col.find(q, {"_id": 1, "file_name": 1}).sort("_id", 1).limit(50)
            new_docs = list(cur)

            if not new_docs:
                time.sleep(CFG.poll_seconds)
                continue

            pending = pending_requests_all()

            for d in new_docs:
                oid = d["_id"]
                fn = d.get("file_name") or ""
                guess_title, guess_year = extract_guess_title_year_from_filename(fn)

                # Attempt to match pending requests
                for r in pending:
                    r_title = r.get("title") or ""
                    r_year = r.get("year")
                    # year strict if request has year
                    if r_year and guess_year and guess_year != r_year:
                        continue

                    sc = score_match(r_title, guess_title, r_year)
                    if r_year:
                        ok = sc >= 0.70
                    else:
                        ok = sc >= 0.80

                    if ok:
                        # mark request done
                        try:
                            requests_col.update_one(
                                {"_id": r["_id"], "status": "pending"},
                                {"$set": {"status": "done", "done_at": now_utc(), "matched_file": fn}}
                            )
                        except Exception:
                            pass

                        # notify user
                        notify_user_available(r["user_id"], r["media_type"], int(r["tmdb_id"]))

                last_oid = oid
                meta_set_last_oid(last_oid)

        except Exception as e:
            log(f"Poller error: {repr(e)}")

        time.sleep(CFG.poll_seconds)

# ============================================================
# MAIN
# ============================================================
def sanity_check() -> List[str]:
    probs = []
    if not CFG.bot_token: probs.append("BOT_TOKEN missing")
    if not CFG.api_hash or CFG.api_id == 0: probs.append("API_ID/API_HASH missing")
    if not CFG.mongo_uri: probs.append("MONGO_URI missing")
    if not CFG.tmdb_api_key: probs.append("TMDB_API_KEY missing")
    if CFG.allowed_group_id == 0: probs.append("ALLOWED_GROUP_ID missing")
    if CFG.admin_req_channel_id == 0: probs.append("ADMIN_REQ_CHANNEL_ID missing")
    return probs

def main():
    # Always start web server so Render sees a port
    threading.Thread(target=run_web, daemon=True).start()

    probs = sanity_check()
    if probs:
        log("CONFIG PROBLEMS: " + ", ".join(probs))
        while True:
            time.sleep(60)

    # init mongo
    try:
        init_mongo()
        if mongo_ping_ok():
            log("MongoDB connected ✅")
        else:
            log("Mongo ping failed")
            while True:
                time.sleep(60)
    except Exception as e:
        log(f"Mongo init failed: {repr(e)}")
        while True:
            time.sleep(60)

    # start poller
    threading.Thread(target=poll_worker, daemon=True).start()

    # run bot
    log("Bot starting...")
    bot.run()

if __name__ == "__main__":
    main()
