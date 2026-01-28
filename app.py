#!/usr/bin/env python3
"""
Ultra Pro Max TMDB AutoFilter Bot
Deploy on Render - Free Tier Compatible
"""

import os
import re
import sys
import time
import math
import json
import signal
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import requests
from flask import Flask, request, jsonify
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError, OperationFailure
from bson import ObjectId, json_util
from werkzeug.middleware.profiler import ProfilerMiddleware

from pyrogram import Client, filters, idle
from pyrogram.errors import (
    FloodWait, RPCError, UserNotParticipant, 
    ChatAdminRequired, ChannelInvalid, ChannelPrivate
)
from pyrogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Message,
    CallbackQuery,
    User,
    Chat
)

# ============================================================
# CONFIGURATION WITH RENDER COMPATIBILITY
# ============================================================
def get_env_str(name: str, default: str = "") -> str:
    """Get environment variable safely"""
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip()

def get_env_int(name: str, default: int = 0) -> int:
    """Get environment variable as integer"""
    try:
        value = os.getenv(name)
        if value is None:
            return default
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default

def get_env_bool(name: str, default: bool = False) -> bool:
    """Get environment variable as boolean"""
    value = os.getenv(name, "").lower().strip()
    if value in ("true", "yes", "1", "on", "y"):
        return True
    elif value in ("false", "no", "0", "off", "n"):
        return False
    return default

@dataclass
class BotConfig:
    # Telegram API
    bot_token: str
    api_id: int
    api_hash: str
    
    # Telegram IDs
    allowed_group_id: int
    admin_req_channel_id: int
    log_channel_id: int
    
    # Database
    mongo_uri: str
    mongo_timeout: int = 10000
    mongo_retry_attempts: int = 3
    
    # APIs
    tmdb_api_key: str
    
    # App Settings
    port: int = 10000
    app_name: str = "ultra-pro-max-bot"
    webhook_url: Optional[str] = None
    
    # Database Names
    files_db_name: str = "autofilter"
    files_collection: str = "royal_files"
    new_db_name: str = "requestbot"
    
    # Limits
    result_buttons: int = 10
    max_requests: int = 3
    max_search_results: int = 20
    poll_seconds: int = 30
    scan_limit: int = 3000
    request_expire_hours: int = 168  # 7 days
    
    # Features
    debug_mode: bool = False
    maintenance_mode: bool = False
    enable_auto_notify: bool = True
    enable_request_system: bool = True
    enable_file_search: bool = True
    
    # Performance
    cache_ttl: int = 300  # 5 minutes
    max_file_cache: int = 1000
    connection_pool_size: int = 10
    
    def validate(self) -> List[str]:
        """Validate configuration and return errors"""
        errors = []
        
        if not self.bot_token or not self.bot_token.startswith(""):
            errors.append("Invalid BOT_TOKEN")
        
        if not self.api_id or self.api_id <= 0:
            errors.append("Invalid API_ID")
        
        if not self.api_hash or len(self.api_hash) < 10:
            errors.append("Invalid API_HASH")
        
        if not self.allowed_group_id:
            errors.append("ALLOWED_GROUP_ID required")
        
        if not self.admin_req_channel_id:
            errors.append("ADMIN_REQ_CHANNEL_ID required")
        
        if not self.mongo_uri or "mongodb" not in self.mongo_uri:
            errors.append("Invalid MONGO_URI")
        
        if not self.tmdb_api_key or len(self.tmdb_api_key) < 10:
            errors.append("Invalid TMDB_API_KEY")
        
        return errors

# Initialize Configuration
CFG = BotConfig(
    # Required - Set in Render Environment Variables
    bot_token=get_env_str("BOT_TOKEN"),
    api_id=get_env_int("API_ID"),
    api_hash=get_env_str("API_HASH"),
    allowed_group_id=get_env_int("ALLOWED_GROUP_ID"),
    admin_req_channel_id=get_env_int("ADMIN_REQ_CHANNEL_ID"),
    log_channel_id=get_env_int("LOG_CHANNEL_ID", 0),
    mongo_uri=get_env_str("MONGO_URI"),
    tmdb_api_key=get_env_str("TMDB_API_KEY"),
    
    # Optional with defaults
    port=get_env_int("PORT", 10000),
    app_name=get_env_str("APP_NAME", "ultra-pro-max-bot"),
    webhook_url=get_env_str("WEBHOOK_URL", ""),
    
    # Features
    debug_mode=get_env_bool("DEBUG_MODE", False),
    maintenance_mode=get_env_bool("MAINTENANCE_MODE", False),
    enable_auto_notify=get_env_bool("ENABLE_AUTO_NOTIFY", True),
    enable_request_system=get_env_bool("ENABLE_REQUEST_SYSTEM", True),
    enable_file_search=get_env_bool("ENABLE_FILE_SEARCH", True),
    
    # Performance
    cache_ttl=get_env_int("CACHE_TTL", 300),
    max_file_cache=get_env_int("MAX_FILE_CACHE", 1000),
    connection_pool_size=get_env_int("CONNECTION_POOL_SIZE", 10),
)

# ============================================================
# ADVANCED LOGGING SYSTEM
# ============================================================
class Logger:
    """Advanced logging system with colors and file output"""
    
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, name="BOT"):
        self.name = name
        self.log_file = "bot.log"
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
    def _write_to_file(self, level: str, message: str):
        """Write log to file with rotation"""
        try:
            # Check if file needs rotation
            if os.path.exists(self.log_file):
                if os.path.getsize(self.log_file) > self.max_file_size:
                    # Rotate log file
                    for i in range(5, 0, -1):
                        old_file = f"{self.log_file}.{i}"
                        new_file = f"{self.log_file}.{i+1}"
                        if os.path.exists(old_file):
                            if os.path.exists(new_file):
                                os.remove(new_file)
                            os.rename(old_file, new_file)
                    os.rename(self.log_file, f"{self.log_file}.1")
            
            # Write log entry
            with open(self.log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{timestamp}] [{level}] {message}\n")
        except Exception:
            pass
    
    def log(self, level: str, message: str, exc_info=None):
        """Log a message with specified level"""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        # Format message
        if exc_info:
            if isinstance(exc_info, Exception):
                message = f"{message} - {type(exc_info).__name__}: {exc_info}"
            else:
                message = f"{message} - {exc_info}"
        
        # Add bot name prefix
        formatted = f"[{timestamp}] [{level:^8}] [{self.name}] {message}"
        
        # Write to file
        self._write_to_file(level, message)
        
        # Print to console with colors if not in production
        if CFG.debug_mode or level in ("ERROR", "CRITICAL", "WARNING"):
            color = self.COLORS.get(level, self.COLORS['RESET'])
            print(f"{color}{formatted}{self.COLORS['RESET']}", flush=True)
        else:
            print(formatted, flush=True)
    
    def debug(self, message: str):
        if CFG.debug_mode:
            self.log("DEBUG", message)
    
    def info(self, message: str):
        self.log("INFO", message)
    
    def warning(self, message: str, exc_info=None):
        self.log("WARNING", message, exc_info)
    
    def error(self, message: str, exc_info=None):
        self.log("ERROR", message, exc_info)
    
    def critical(self, message: str, exc_info=None):
        self.log("CRITICAL", message, exc_info)
        # In critical errors, also send to Telegram if log channel is set
        if CFG.log_channel_id:
            try:
                bot = get_bot()
                if bot and bot.is_connected:
                    asyncio.run_coroutine_threadsafe(
                        bot.send_message(
                            CFG.log_channel_id,
                            f"🚨 CRITICAL: {message[:1000]}"
                        ),
                        bot.loop
                    )
            except Exception:
                pass

# Global logger instance
log = Logger("MAIN")

# ============================================================
# FLASK WEB SERVER (RENDER COMPATIBLE)
# ============================================================
web = Flask(__name__)

# Health check endpoint for Render
@web.route("/")
def home():
    """Health check endpoint"""
    status = {
        "status": "running",
        "service": "Ultra Pro Max Bot",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": get_uptime(),
        "config": {
            "debug_mode": CFG.debug_mode,
            "maintenance_mode": CFG.maintenance_mode,
            "features": {
                "auto_notify": CFG.enable_auto_notify,
                "request_system": CFG.enable_request_system,
                "file_search": CFG.enable_file_search
            }
        }
    }
    return jsonify(status)

@web.route("/health")
def health():
    """Detailed health check"""
    try:
        # Check MongoDB
        mongo_status = "healthy" if mongo_ping() else "unhealthy"
        
        # Check TMDB
        tmdb_status = "unknown"
        try:
            response = requests.get(
                f"https://api.themoviedb.org/3/configuration",
                params={"api_key": CFG.tmdb_api_key},
                timeout=5
            )
            tmdb_status = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            tmdb_status = "unhealthy"
        
        # Check bot status
        bot_status = "unknown"
        try:
            bot = get_bot()
            bot_status = "connected" if bot and bot.is_connected else "disconnected"
        except:
            bot_status = "error"
        
        return jsonify({
            "status": "healthy",
            "checks": {
                "mongodb": mongo_status,
                "tmdb": tmdb_status,
                "bot": bot_status
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@web.route("/stats")
def stats():
    """Get bot statistics"""
    try:
        stats_data = {
            "users_count": users_col.count_documents({}) if users_col else 0,
            "files_count": files_col.count_documents({}) if files_col else 0,
            "pending_requests": requests_col.count_documents({"status": "pending"}) if requests_col else 0,
            "total_requests": requests_col.count_documents({}) if requests_col else 0,
            "completed_requests": requests_col.count_documents({"status": "done"}) if requests_col else 0,
            "uptime": get_uptime(),
            "memory_usage": get_memory_usage(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return jsonify(stats_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web.route("/webhook", methods=["POST"])
def webhook():
    """Webhook endpoint for external services"""
    if CFG.webhook_url:
        try:
            data = request.json
            log.info(f"Webhook received: {data}")
            return jsonify({"status": "received"})
        except Exception as e:
            log.error(f"Webhook error: {e}")
            return jsonify({"error": str(e)}), 400
    return jsonify({"error": "Webhook not configured"}), 404

def run_flask():
    """Run Flask web server"""
    try:
        log.info(f"Starting Flask server on port {CFG.port}")
        web.run(
            host="0.0.0.0",
            port=CFG.port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        log.critical(f"Flask server failed: {e}")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
_start_time = time.time()

def get_uptime() -> str:
    """Get formatted uptime"""
    seconds = int(time.time() - _start_time)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def get_memory_usage() -> Dict[str, float]:
    """Get memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0}

def format_bytes(size: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert to integer"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_str(value: Any, default: str = "") -> str:
    """Safely convert to string"""
    if value is None:
        return default
    return str(value).strip()

# ============================================================
# DATABASE CONNECTION WITH RETRY LOGIC
# ============================================================
class MongoDBManager:
    """Managed MongoDB connection with retry logic"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.last_connection_attempt = None
        self.connection_errors = 0
        self.max_retries = CFG.mongo_retry_attempts
        
    def connect(self) -> bool:
        """Establish MongoDB connection with retry"""
        if self.connected and self.client:
            try:
                self.client.admin.command('ping')
                return True
            except:
                self.connected = False
        
        for attempt in range(self.max_retries):
            try:
                log.info(f"Connecting to MongoDB (attempt {attempt + 1}/{self.max_retries})")
                
                # Connection options for Render compatibility
                self.client = MongoClient(
                    CFG.mongo_uri,
                    connectTimeoutMS=CFG.mongo_timeout,
                    serverSelectionTimeoutMS=CFG.mongo_timeout,
                    socketTimeoutMS=CFG.mongo_timeout,
                    maxPoolSize=CFG.connection_pool_size,
                    minPoolSize=1,
                    retryWrites=True,
                    retryReads=True,
                    appname=CFG.app_name
                )
                
                # Test connection
                self.client.admin.command('ping')
                
                # Initialize collections
                self._init_collections()
                
                self.connected = True
                self.connection_errors = 0
                self.last_connection_attempt = datetime.now(timezone.utc)
                
                log.info("MongoDB connected successfully")
                return True
                
            except Exception as e:
                self.connection_errors += 1
                log.error(f"MongoDB connection failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    log.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    log.critical(f"Failed to connect to MongoDB after {self.max_retries} attempts")
                    self.connected = False
                    return False
        
        return False
    
    def _init_collections(self):
        """Initialize database collections"""
        global files_col, users_col, requests_col, meta_col, stats_col, cache_col
        
        try:
            # Main files collection (read-only for autofilter)
            files_db = self.client[CFG.files_db_name]
            files_col = files_db[CFG.files_collection]
            
            # Bot database
            bot_db = self.client[CFG.new_db_name]
            
            # Users collection
            users_col = bot_db["users"]
            users_col.create_index([("user_id", 1)], unique=True)
            users_col.create_index([("last_seen", -1)])
            users_col.create_index([("requests_count", -1)])
            
            # Requests collection
            requests_col = bot_db["requests"]
            requests_col.create_index([("user_id", 1), ("status", 1)])
            requests_col.create_index([("tmdb_id", 1), ("media_type", 1), ("status", 1)])
            requests_col.create_index([("status", 1), ("created_at", -1)])
            requests_col.create_index([("created_at", -1)])
            
            # Metadata collection
            meta_col = bot_db["meta"]
            meta_col.create_index([("key", 1)], unique=True)
            
            # Statistics collection
            stats_col = bot_db["stats"]
            stats_col.create_index([("date", 1)], unique=True)
            
            # Cache collection
            cache_col = bot_db["cache"]
            cache_col.create_index([("key", 1)], unique=True)
            cache_col.create_index([("expires_at", 1)], expireAfterSeconds=0)
            
            log.info("Database collections initialized")
            
        except OperationFailure as e:
            # If we can't create indexes (Atlas free tier), just log and continue
            log.warning(f"Could not create indexes (might be Atlas free tier): {e}")
            # Still assign collections
            files_db = self.client[CFG.files_db_name]
            files_col = files_db[CFG.files_collection]
            
            bot_db = self.client[CFG.new_db_name]
            users_col = bot_db["users"]
            requests_col = bot_db["requests"]
            meta_col = bot_db["meta"]
            stats_col = bot_db["stats"]
            cache_col = bot_db["cache"]
            
        except Exception as e:
            log.error(f"Error initializing collections: {e}")
            raise
    
    def disconnect(self):
        """Close MongoDB connection"""
        try:
            if self.client:
                self.client.close()
                self.connected = False
                log.info("MongoDB disconnected")
        except Exception as e:
            log.error(f"Error disconnecting MongoDB: {e}")

# Global MongoDB manager
mongo_manager = MongoDBManager()

def init_mongo():
    """Initialize MongoDB connection"""
    return mongo_manager.connect()

def mongo_ping() -> bool:
    """Ping MongoDB to check connection"""
    try:
        if mongo_manager.connected and mongo_manager.client:
            mongo_manager.client.admin.command('ping')
            return True
    except Exception:
        pass
    
    # Try to reconnect if disconnected
    if not mongo_manager.connected:
        return mongo_manager.connect()
    
    return False

# ============================================================
# CACHE SYSTEM
# ============================================================
class Cache:
    """Simple cache system using MongoDB"""
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from cache"""
        try:
            if not mongo_ping():
                return default
            
            doc = cache_col.find_one({"key": key})
            if doc:
                expires_at = doc.get("expires_at")
                if expires_at and expires_at < datetime.now(timezone.utc):
                    # Expired, delete it
                    cache_col.delete_one({"_id": doc["_id"]})
                    return default
                return doc.get("value")
        except Exception as e:
            log.debug(f"Cache get error: {e}")
        return default
    
    @staticmethod
    def set(key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL"""
        try:
            if not mongo_ping():
                return
            
            expires_at = None
            if ttl:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            
            cache_col.update_one(
                {"key": key},
                {
                    "$set": {
                        "value": value,
                        "expires_at": expires_at,
                        "updated_at": datetime.now(timezone.utc)
                    },
                    "$setOnInsert": {"created_at": datetime.now(timezone.utc)}
                },
                upsert=True
            )
        except Exception as e:
            log.debug(f"Cache set error: {e}")
    
    @staticmethod
    def delete(key: str):
        """Delete key from cache"""
        try:
            if mongo_ping():
                cache_col.delete_one({"key": key})
        except Exception:
            pass
    
    @staticmethod
    def clear():
        """Clear all cache"""
        try:
            if mongo_ping():
                cache_col.delete_many({})
        except Exception:
            pass

# ============================================================
# TMDB API CLIENT WITH CACHING
# ============================================================
class TMDBClient:
    """TMDB API client with caching and rate limiting"""
    
    BASE_URL = "https://api.themoviedb.org/3"
    DEFAULT_LANGUAGE = "en-US"
    
    def __init__(self):
        self.api_key = CFG.tmdb_api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"{CFG.app_name}/2.0",
            "Accept": "application/json"
        })
        self.rate_limit_remaining = 40
        self.rate_limit_reset = time.time() + 10
        
    def _make_request(self, path: str, params: Optional[dict] = None) -> dict:
        """Make request to TMDB API with rate limiting"""
        # Rate limiting
        current_time = time.time()
        if self.rate_limit_remaining <= 0 and current_time < self.rate_limit_reset:
            wait_time = self.rate_limit_reset - current_time
            log.warning(f"TMDB rate limit hit, waiting {wait_time:.1f}s")
            time.sleep(wait_time + 0.5)
        
        # Prepare request
        url = f"{self.BASE_URL}{path}"
        params = params or {}
        params["api_key"] = self.api_key
        params.setdefault("language", self.DEFAULT_LANGUAGE)
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            # Update rate limit info
            self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 40))
            self.rate_limit_reset = float(response.headers.get('X-RateLimit-Reset', time.time() + 10))
            
            return response.json()
            
        except requests.exceptions.Timeout:
            log.error("TMDB request timeout")
            raise
        except requests.exceptions.RequestException as e:
            log.error(f"TMDB request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            log.error(f"TMDB JSON decode error: {e}")
            raise
    
    def search_multi(self, query: str, page: int = 1) -> List[dict]:
        """Search movies and TV shows"""
        cache_key = f"tmdb_search:{query.lower()}:{page}"
        cached = Cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            data = self._make_request("/search/multi", {
                "query": query,
                "include_adult": "false",
                "page": page
            })
            
            # Filter only movies and TV shows
            results = []
            for item in data.get("results", []):
                media_type = item.get("media_type")
                if media_type in ("movie", "tv"):
                    results.append(item)
            
            # Cache for 1 hour
            Cache.set(cache_key, results, ttl=3600)
            return results
            
        except Exception as e:
            log.error(f"TMDB search error: {e}")
            return []
    
    def get_details(self, media_type: str, tmdb_id: int) -> Optional[dict]:
        """Get full details with credits, external IDs, etc."""
        cache_key = f"tmdb_details:{media_type}:{tmdb_id}"
        cached = Cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Append all possible details
            append_to_response = "credits,external_ids,images,videos,content_ratings,release_dates,similar"
            data = self._make_request(f"/{media_type}/{tmdb_id}", {
                "append_to_response": append_to_response
            })
            
            # Cache for 6 hours
            Cache.set(cache_key, data, ttl=21600)
            return data
            
        except Exception as e:
            log.error(f"TMDB details error: {e}")
            return None
    
    def get_configuration(self) -> Optional[dict]:
        """Get TMDB configuration"""
        cache_key = "tmdb_configuration"
        cached = Cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            data = self._make_request("/configuration")
            Cache.set(cache_key, data, ttl=86400)  # 24 hours
            return data
        except Exception:
            return None
    
    def get_poster_url(self, poster_path: Optional[str], size: str = "w500") -> Optional[str]:
        """Get full poster URL"""
        if not poster_path:
            return None
        
        config = self.get_configuration()
        if config and "images" in config:
            base_url = config["images"]["secure_base_url"]
            return f"{base_url}{size}{poster_path}"
        
        # Fallback
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"

# Global TMDB client
tmdb_client = TMDBClient()

# ============================================================
# TEXT NORMALIZATION AND FUZZY MATCHING (ENHANCED)
# ============================================================
# Regular expressions for cleaning
YEAR_PATTERN = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')
RESOLUTION_PATTERN = re.compile(r'\b(480p|720p|1080p|2160p|4k|8k)\b', re.IGNORECASE)
CODEC_PATTERN = re.compile(r'\b(x264|x265|h264|h265|hevc|avc|av1)\b', re.IGNORECASE)
AUDIO_PATTERN = re.compile(r'\b(aac|ac3|dd|ddp|dts|eac3|atmos)\b', re.IGNORECASE)
SOURCE_PATTERN = re.compile(r'\b(webrip|webdl|web-dl|bluray|brrip|dvdrip|hdtv|pdtv)\b', re.IGNORECASE)

# Comprehensive junk words list for Sinhala/English movie files
JUNK_WORDS = {
    # Quality/Resolution
    "480p", "720p", "1080p", "2160p", "4k", "8k", "hdr", "sdr", "uhd", "fhd", "hd", "sd",
    "10bit", "8bit", "hdr10", "hdr10plus", "dv", "dolbyvision",
    
    # Codec/Video
    "x264", "x265", "h264", "h265", "hevc", "av1", "avc", "divx", "xvid",
    
    # Audio
    "aac", "ac3", "dd", "ddp", "dts", "eac3", "atmos", "truehd", "mp3",
    
    # Source
    "webrip", "webdl", "web-dl", "bluray", "brrip", "dvdrip", "hdtv", "pdtv", 
    "camrip", "ts", "telesync", "tc", "telecine", "scr", "screener", "dvdscr", "r5",
    "remux", "bdrip", "microhd", "complete", "full",
    
    # Release groups
    "proper", "repack", "rerip", "nf", "amzn", "dsnp", "hulu", "atvp",
    
    # Language/Subtitles
    "sinhala", "sinhalese", "tamil", "telugu", "hindi", "malayalam", "kannada", 
    "english", "dubbed", "dubbing", "dual", "multi", "sub", "subs", "subtitle", 
    "subtitles", "embedded", "softsubs", "hardsub",
    
    # Common words in movie titles (to keep)
    # These are NOT junk - they're part of actual titles
    # "the", "a", "an", "and", "of", "in", "to", "for" - We'll keep these
    
    # Website/Channel tags
    "www", "com", "net", "org", "lk", "in", "to", "me", "co", "uk", "us",
    "cinesubz", "royalmovies", "royalseries", "mlwbd", "mkvcinemas", "moviezworld",
    "desiscandal", "khatrimaza", "worldfree4u", "bollyshare", "pagalmovies",
    "tamilrockers", "isaimini", "madrasrockers", "todaypk", "moviesda",
    "tamilyogi", "movieverse", "moviezindagi", "hdmovieshub", "skymovieshd",
    "yts", "rarbg", "ettv", "etrg", "ctrlhd", "framestor", "tigole",
    
    # General noise
    "channel", "upload", "uploaded", "by", "from", "with", "latest", "new",
    "episode", "episodes", "season", "seasons", "series", "part", "volume",
    "collection", "edition", "version", "uncut", "uncensored", "directors",
    "extended", "unrated", "final", "complete", "full", "movie", "film",
    "theatrical", "cut", "limited", "special", "anniversary",
}

# Words that should be kept even if short
KEEP_WORDS = {
    "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",  # Roman numerals
    "tv", "us", "uk", "eu",  # Country codes
    "3d", "2d",  # 3D/2D
}

def normalize_filename(filename: str) -> Tuple[str, List[str], Optional[str]]:
    """
    Normalize filename for matching.
    Returns: (normalized_string, tokens_list, year_or_none)
    """
    if not filename:
        return "", [], None
    
    # Convert to lowercase
    text = filename.lower().strip()
    
    # Remove common prefixes and suffixes from Sri Lankan channels
    # Pattern: [Group] Movie.Name.Year.Quality.mkv
    patterns_to_remove = [
        # Remove [GROUP] prefix
        r'^\[[^\]]+\]\s*',
        # Remove (YEAR) suffix
        r'\s*\(\d{4}\)\s*$',
        # Remove .YEAR. quality groups
        r'\.(19|20)\d{2}\.(480|720|1080|2160)p',
        # Remove file extensions
        r'\.(mkv|mp4|avi|mov|wmv|flv|webm|m4v|ts)$',
        # Remove website URLs
        r'[\w\-]+\.(com|net|org|in|lk|to|me)\b',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Extract year
    year_match = YEAR_PATTERN.search(text)
    year = year_match.group(1) if year_match else None
    
    # Remove year from text for further processing
    if year:
        text = YEAR_PATTERN.sub(' ', text)
    
    # Replace separators with spaces
    text = re.sub(r'[\._\-\[\]\(\)\{\}]', ' ', text)
    
    # Remove special characters but keep letters, numbers, and spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split into tokens
    tokens = text.split()
    
    # Filter tokens
    filtered_tokens = []
    for token in tokens:
        # Skip if token is in junk words
        if token in JUNK_WORDS:
            continue
        
        # Keep certain words even if short
        if token in KEEP_WORDS:
            filtered_tokens.append(token)
            continue
        
        # Skip single characters (unless they're part of Roman numerals)
        if len(token) == 1 and token not in 'ivx':
            continue
        
        # Skip pure numbers (except years which we already extracted)
        if token.isdigit() and len(token) != 4:
            continue
        
        # Skip tokens that look like quality indicators
        if RESOLUTION_PATTERN.match(token) or CODEC_PATTERN.match(token) or \
           AUDIO_PATTERN.match(token) or SOURCE_PATTERN.match(token):
            continue
        
        filtered_tokens.append(token)
    
    # Create normalized string
    normalized = ' '.join(filtered_tokens)
    
    return normalized, filtered_tokens, year

def calculate_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles using multiple algorithms"""
    
    # If either is empty, return 0
    if not title1 or not title2:
        return 0.0
    
    # Direct equality (case insensitive)
    if title1.lower() == title2.lower():
        return 1.0
    
    # SequenceMatcher ratio
    seq_ratio = SequenceMatcher(None, title1.lower(), title2.lower()).ratio()
    
    # Token set similarity (Jaccard)
    tokens1 = set(title1.lower().split())
    tokens2 = set(title2.lower().split())
    
    if not tokens1 or not tokens2:
        jaccard = 0.0
    else:
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        jaccard = intersection / union if union > 0 else 0.0
    
    # Weighted average
    similarity = (0.4 * seq_ratio) + (0.6 * jaccard)
    
    # Bonus for partial matches (one contains the other)
    if title1.lower() in title2.lower() or title2.lower() in title1.lower():
        similarity = min(1.0, similarity + 0.1)
    
    return similarity

def find_matching_files(tmdb_title: str, year: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find matching files in database using fuzzy matching.
    Returns list of dicts with file_name and similarity_score.
    """
    if not CFG.enable_file_search or not files_col:
        return []
    
    # Normalize the TMDB title
    norm_title, title_tokens, _ = normalize_filename(tmdb_title)
    if not title_tokens:
        return []
    
    # Use the first token for initial filtering (for performance)
    first_token = title_tokens[0] if title_tokens else ""
    if not first_token or len(first_token) < 2:
        return []
    
    try:
        # Query for files containing the first token
        query = {"file_name": {"$regex": re.escape(first_token), "$options": "i"}}
        
        matches = []
        cursor = files_col.find(query, {"file_name": 1}).limit(CFG.scan_limit)
        
        for doc in cursor:
            filename = doc.get("file_name", "")
            if not filename:
                continue
            
            # Calculate similarity
            similarity = calculate_similarity(norm_title, filename)
            
            # Apply year filter if available
            if year:
                # Extract year from filename
                year_match = YEAR_PATTERN.search(filename)
                if year_match:
                    file_year = year_match.group(1)
                    if file_year != year:
                        # Different year - penalize heavily
                        similarity *= 0.3
            
            # Apply threshold
            if similarity >= 0.6:  # Adjustable threshold
                matches.append({
                    "file_name": filename,
                    "score": similarity,
                    "year": year_match.group(1) if year_match else None
                })
        
        # Sort by score (descending)
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Remove duplicates and limit results
        unique_matches = []
        seen = set()
        
        for match in matches:
            if match["file_name"] not in seen:
                seen.add(match["file_name"])
                unique_matches.append(match)
                
                if len(unique_matches) >= limit:
                    break
        
        return unique_matches
        
    except Exception as e:
        log.error(f"Error finding matching files: {e}")
        return []

# ============================================================
# REQUEST MANAGEMENT SYSTEM
# ============================================================
class RequestManager:
    """Manage user requests with expiration and limits"""
    
    @staticmethod
    def now() -> datetime:
        """Get current UTC datetime"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def get_user_requests_count(user_id: int) -> int:
        """Get count of pending requests for a user"""
        try:
            if not mongo_ping():
                return 0
            
            return requests_col.count_documents({
                "user_id": user_id,
                "status": "pending",
                "created_at": {"$gte": RequestManager.now() - timedelta(hours=CFG.request_expire_hours)}
            })
        except Exception as e:
            log.error(f"Error getting user requests count: {e}")
            return 0
    
    @staticmethod
    def get_user_requests(user_id: int, limit: int = 10) -> List[Dict]:
        """Get list of user's pending requests"""
        try:
            if not mongo_ping():
                return []
            
            return list(requests_col.find({
                "user_id": user_id,
                "status": "pending"
            }).sort("created_at", -1).limit(limit))
        except Exception as e:
            log.error(f"Error getting user requests: {e}")
            return []
    
    @staticmethod
    def create_request(
        user_id: int, 
        media_type: str, 
        tmdb_id: int, 
        title: str, 
        year: Optional[str] = None
    ) -> bool:
        """Create a new request"""
        try:
            if not mongo_ping():
                return False
            
            # Check if already exists (pending)
            existing = requests_col.find_one({
                "user_id": user_id,
                "media_type": media_type,
                "tmdb_id": tmdb_id,
                "status": "pending"
            })
            
            if existing:
                return True  # Already exists
            
            # Check limit
            if RequestManager.get_user_requests_count(user_id) >= CFG.max_requests:
                return False
            
            # Create request
            request_data = {
                "user_id": user_id,
                "media_type": media_type,
                "tmdb_id": tmdb_id,
                "title": title,
                "year": year,
                "status": "pending",
                "created_at": RequestManager.now(),
                "updated_at": RequestManager.now()
            }
            
            requests_col.insert_one(request_data)
            
            # Update user stats
            users_col.update_one(
                {"user_id": user_id},
                {
                    "$inc": {"requests_count": 1},
                    "$set": {"last_request": RequestManager.now()},
                    "$setOnInsert": {
                        "user_id": user_id,
                        "first_seen": RequestManager.now()
                    }
                },
                upsert=True
            )
            
            # Update daily stats
            today = RequestManager.now().date().isoformat()
            stats_col.update_one(
                {"date": today},
                {
                    "$inc": {"requests_created": 1},
                    "$setOnInsert": {"date": today}
                },
                upsert=True
            )
            
            return True
            
        except Exception as e:
            log.error(f"Error creating request: {e}")
            return False
    
    @staticmethod
    def cancel_request(request_id: str, user_id: int) -> bool:
        """Cancel a request"""
        try:
            if not mongo_ping():
                return False
            
            result = requests_col.update_one(
                {
                    "_id": ObjectId(request_id),
                    "user_id": user_id,
                    "status": "pending"
                },
                {
                    "$set": {
                        "status": "cancelled",
                        "cancelled_at": RequestManager.now(),
                        "updated_at": RequestManager.now()
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            log.error(f"Error cancelling request: {e}")
            return False
    
    @staticmethod
    def mark_requests_completed(media_type: str, tmdb_id: int) -> int:
        """Mark all pending requests for a media as completed"""
        try:
            if not mongo_ping():
                return 0
            
            result = requests_col.update_many(
                {
                    "media_type": media_type,
                    "tmdb_id": tmdb_id,
                    "status": "pending"
                },
                {
                    "$set": {
                        "status": "done",
                        "done_at": RequestManager.now(),
                        "updated_at": RequestManager.now()
                    }
                }
            )
            
            count = result.modified_count
            if count > 0:
                log.info(f"Marked {count} requests as completed for {media_type} {tmdb_id}")
            
            return count
            
        except Exception as e:
            log.error(f"Error marking requests completed: {e}")
            return 0
    
    @staticmethod
    def cleanup_expired_requests():
        """Clean up expired requests"""
        try:
            if not mongo_ping():
                return
            
            cutoff = RequestManager.now() - timedelta(hours=CFG.request_expire_hours)
            
            result = requests_col.update_many(
                {
                    "status": "pending",
                    "created_at": {"$lt": cutoff}
                },
                {
                    "$set": {
                        "status": "expired",
                        "expired_at": RequestManager.now(),
                        "updated_at": RequestManager.now()
                    }
                }
            )
            
            if result.modified_count > 0:
                log.info(f"Cleaned up {result.modified_count} expired requests")
                
        except Exception as e:
            log.error(f"Error cleaning up expired requests: {e}")

# ============================================================
# TELEGRAM BOT SETUP
# ============================================================
# Global bot instance
bot = None
bot_username = None

def get_bot():
    """Get the bot instance (singleton pattern)"""
    global bot
    return bot

async def get_bot_username_safe() -> str:
    """Get bot username safely"""
    global bot_username
    if bot_username:
        return bot_username
    
    try:
        bot_instance = get_bot()
        if bot_instance and bot_instance.is_connected:
            me = await bot_instance.get_me()
            bot_username = me.username
            return bot_username
    except Exception as e:
        log.error(f"Error getting bot username: {e}")
    
    return "unknown_bot"

# ============================================================
# KEYBOARD BUILDERS
# ============================================================
class KeyboardBuilder:
    """Build inline keyboards for various purposes"""
    
    @staticmethod
    def start_pm_keyboard() -> InlineKeyboardMarkup:
        """Keyboard for starting bot in PM"""
        username = asyncio.run(get_bot_username_safe())
        button = InlineKeyboardButton(
            "🔓 Start Bot in PM",
            url=f"https://t.me/{username}?start=start"
        )
        return InlineKeyboardMarkup([[button]])
    
    @staticmethod
    def search_results_keyboard(results: List[Dict], page: int = 1) -> InlineKeyboardMarkup:
        """Keyboard for search results"""
        buttons = []
        
        for result in results[:CFG.result_buttons]:
            media_type = result.get("media_type", "movie")
            tmdb_id = result.get("id", 0)
            
            # Get title
            if media_type == "movie":
                title = result.get("title", "Unknown")
            else:
                title = result.get("name", "Unknown")
            
            # Get year
            date = result.get("release_date") if media_type == "movie" else result.get("first_air_date")
            year = date[:4] if date else "----"
            
            # Truncate title if too long
            if len(title) > 35:
                title = title[:32] + "..."
            
            # Create button
            icon = "🎬" if media_type == "movie" else "📺"
            button_text = f"{icon} {title} ({year})"
            
            buttons.append([InlineKeyboardButton(
                button_text,
                callback_data=f"detail:{media_type}:{tmdb_id}:{page}"
            )])
        
        # Add navigation if needed (simplified for now)
        if len(results) > CFG.result_buttons:
            buttons.append([
                InlineKeyboardButton("⬅️ Previous", callback_data=f"page:{page-1}"),
                InlineKeyboardButton(f"Page {page}", callback_data="noop"),
                InlineKeyboardButton("Next ➡️", callback_data=f"page:{page+1}")
            ])
        
        return InlineKeyboardMarkup(buttons)
    
    @staticmethod
    def detail_keyboard(media_type: str, tmdb_id: int) -> InlineKeyboardMarkup:
        """Keyboard for detail view"""
        buttons = []
        
        if CFG.enable_request_system:
            buttons.append([InlineKeyboardButton(
                "📥 Request / ඉල්ලන්න",
                callback_data=f"request:{media_type}:{tmdb_id}"
            )])
        
        buttons.append([InlineKeyboardButton(
            "🔍 Search Again",
            switch_inline_query_current_chat=""
        )])
        
        return InlineKeyboardMarkup(buttons)
    
    @staticmethod
    def request_management_keyboard(requests: List[Dict]) -> InlineKeyboardMarkup:
        """Keyboard for managing requests"""
        buttons = []
        
        for req in requests[:CFG.max_requests]:
            req_id = str(req.get("_id", ""))
            title = req.get("title", "Unknown")
            year = req.get("year", "")
            
            # Truncate if too long
            button_text = f"🗑 {title}"
            if year:
                button_text += f" ({year})"
            
            if len(button_text) > 30:
                button_text = button_text[:27] + "..."
            
            buttons.append([InlineKeyboardButton(
                button_text,
                callback_data=f"cancel_request:{req_id}"
            )])
        
        # Add close button
        if buttons:
            buttons.append([InlineKeyboardButton(
                "❌ Close",
                callback_data="close"
            )])
        
        return InlineKeyboardMarkup(buttons)
    
    @staticmethod
    def admin_actions_keyboard(media_type: str, tmdb_id: int, user_id: int) -> InlineKeyboardMarkup:
        """Keyboard for admin actions"""
        buttons = [
            [
                InlineKeyboardButton(
                    "✅ Mark as Filled",
                    callback_data=f"admin_fill:{media_type}:{tmdb_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    "❌ Cancel User Request",
                    callback_data=f"admin_cancel:{media_type}:{tmdb_id}:{user_id}"
                )
            ]
        ]
        return InlineKeyboardMarkup(buttons)

# ============================================================
# MESSAGE BUILDERS (RICH CARDS)
# ============================================================
class MessageBuilder:
    """Build rich messages/cards for the bot"""
    
    @staticmethod
    def build_start_message() -> str:
        """Build start message"""
        return (
            "👋 **Welcome to Ultra Pro Max Movie Finder!**\n\n"
            "🎬 **Features:**\n"
            "• Search movies/TV shows in group\n"
            "• Get detailed info in PM\n"
            "• Check availability in our database\n"
            "• Request unavailable content\n"
            "• Get notified when available\n\n"
            "⚡ **How to use:**\n"
            "1. Go to the authorized group\n"
            "2. Type a movie/series name\n"
            "3. Select from results\n"
            "4. View details in PM\n"
            "5. Request if not available\n\n"
            "📊 **Limits:**\n"
            f"• Max requests: {CFG.max_requests} per user\n"
            f"• Request expires in: {CFG.request_expire_hours // 24} days\n\n"
            "🔧 **Support:**\n"
            "Contact @admin for help\n\n"
            "🚀 **සිංහලෙන්:**\n"
            "Group එකේ නමක් type කරන්න → Buttons එයි → Select කරන්න → "
            "PM එකට details එයි → තියෙනවද check කරන්න → නැත්තම් request කරන්න"
        )
    
    @staticmethod
    def build_detail_card(media_data: Dict, media_type: str) -> Tuple[str, Optional[str]]:
        """Build detailed info card (IMDb style)"""
        
        if media_type == "movie":
            title = media_data.get("title", "Unknown")
            date = media_data.get("release_date", "")
            runtime = media_data.get("runtime", 0)
            tagline = media_data.get("tagline", "")
        else:
            title = media_data.get("name", "Unknown")
            date = media_data.get("first_air_date", "")
            runtime = None
            tagline = ""
        
        # Basic info
        year = date[:4] if date else "N/A"
        rating = media_data.get("vote_average", 0)
        vote_count = media_data.get("vote_count", 0)
        status = media_data.get("status", "N/A")
        
        # Genres
        genres = ", ".join([g["name"] for g in media_data.get("genres", [])]) or "N/A"
        
        # Overview
        overview = media_data.get("overview", "No overview available.")
        
        # Credits
        credits = media_data.get("credits", {})
        
        if media_type == "movie":
            # Director for movies
            director = "N/A"
            crew = credits.get("crew", [])
            for person in crew:
                if person.get("job") == "Director":
                    director = person.get("name", "N/A")
                    break
        else:
            # Creator for TV shows
            creator = "N/A"
            created_by = media_data.get("created_by", [])
            if created_by:
                creator = created_by[0].get("name", "N/A")
        
        # Cast (top 5)
        cast = credits.get("cast", [])
        cast_names = [c.get("name", "") for c in cast[:5] if c.get("name")]
        cast_text = ", ".join(cast_names) if cast_names else "N/A"
        
        # External IDs
        external_ids = media_data.get("external_ids", {})
        imdb_id = external_ids.get("imdb_id")
        
        # TMDB link
        tmdb_id = media_data.get("id", 0)
        tmdb_link = f"https://www.themoviedb.org/{media_type}/{tmdb_id}"
        
        # Poster
        poster_path = media_data.get("poster_path")
        poster_url = tmdb_client.get_poster_url(poster_path) if poster_path else None
        
        # Build message
        lines = []
        
        # Title
        lines.append(f"🎬 **{title}** ({year})")
        
        if tagline:
            lines.append(f"*{tagline}*")
        
        lines.append("")  # Empty line
        
        # Rating
        rating_text = f"⭐ **{rating:.1f}/10**" if rating > 0 else "⭐ No rating"
        lines.append(f"{rating_text} • 📊 {vote_count:,} votes")
        
        # Genres
        lines.append(f"🎭 **Genres:** {genres}")
        
        # Status
        lines.append(f"📌 **Status:** {status}")
        
        # Runtime or Seasons
        if media_type == "movie":
            if runtime:
                hours = runtime // 60
                minutes = runtime % 60
                runtime_text = f"{hours}h {minutes}m" if hours else f"{minutes}m"
                lines.append(f"⏱ **Runtime:** {runtime_text}")
        else:
            seasons = media_data.get("number_of_seasons", 0)
            episodes = media_data.get("number_of_episodes", 0)
            lines.append(f"📺 **Seasons:** {seasons} • **Episodes:** {episodes}")
        
        # Director/Creator
        if media_type == "movie":
            lines.append(f"🎬 **Director:** {director}")
        else:
            lines.append(f"👨‍💼 **Creator:** {creator}")
        
        # Cast
        lines.append(f"👥 **Cast:** {cast_text}")
        
        lines.append("")  # Empty line
        
        # Overview
        lines.append("**📝 Overview:**")
        lines.append(overview[:500] + ("..." if len(overview) > 500 else ""))
        
        lines.append("")  # Empty line
        
        # IDs and links
        if imdb_id:
            lines.append(f"🎭 **IMDb ID:** `{imdb_id}`")
        
        lines.append(f"🔗 **TMDB:** {tmdb_link}")
        
        return "\n".join(lines), poster_url
    
    @staticmethod
    def build_availability_message(
        media_title: str,
        year: Optional[str],
        matches: List[Dict],
        is_available: bool
    ) -> str:
        """Build availability message with file list"""
        
        if is_available and matches:
            # Available
            lines = [
                "✅ **Available in our database!**",
                "",
                "📁 **Matching files:**",
            ]
            
            for i, match in enumerate(matches[:5], 1):
                filename = match.get("file_name", "Unknown")
                score = match.get("score", 0)
                
                # Truncate long filenames
                if len(filename) > 60:
                    filename = filename[:57] + "..."
                
                lines.append(f"{i}. `{filename}`")
                if CFG.debug_mode:
                    lines[-1] += f" _(score: {score:.2f})_"
            
            if len(matches) > 5:
                lines.append(f"\n*... and {len(matches) - 5} more files*")
            
            lines.append("\n👉 Search in the group to get download links.")
            
        else:
            # Not available
            lines = [
                "❌ **Not available in our database.**",
                "",
                "💡 You can request this content using the button below.",
                "We'll notify you when it becomes available.",
                "",
                "⚠️ **Note:** Requesting doesn't guarantee fulfillment.",
                "It depends on availability and uploader's discretion."
            ]
        
        return "\n".join(lines)

# ============================================================
# BOT COMMAND HANDLERS
# ============================================================
async def register_user(user_id: int, username: Optional[str], first_name: Optional[str]):
    """Register or update user in database"""
    try:
        if not mongo_ping():
            return
        
        update_data = {
            "username": username,
            "first_name": first_name,
            "last_seen": datetime.now(timezone.utc),
            "$inc": {"messages_count": 1}
        }
        
        users_col.update_one(
            {"user_id": user_id},
            {
                "$set": update_data,
                "$setOnInsert": {
                    "user_id": user_id,
                    "first_seen": datetime.now(timezone.utc),
                    "requests_count": 0,
                    "messages_count": 1
                }
            },
            upsert=True
        )
        
    except Exception as e:
        log.error(f"Error registering user: {e}")

@bot.on_message(filters.command("start") & filters.private)
async def start_command(client: Client, message: Message):
    """Handle /start command"""
    user = message.from_user
    
    # Register user
    await register_user(user.id, user.username, user.first_name)
    
    # Send welcome message
    await message.reply_text(
        MessageBuilder.build_start_message(),
        reply_markup=KeyboardBuilder.start_pm_keyboard(),
        disable_web_page_preview=True
    )

@bot.on_message(filters.command("help") & filters.private)
async def help_command(client: Client, message: Message):
    """Handle /help command"""
    help_text = (
        "🆘 **Help Guide**\n\n"
        
        "🎯 **How to search:**\n"
        "1. Go to the authorized group\n"
        "2. Type movie/series name\n"
        "3. Select from results\n"
        "4. View details in PM\n\n"
        
        "📥 **How to request:**\n"
        "1. Search for content\n"
        "2. If not available, click 'Request' button\n"
        "3. Wait for notification when available\n\n"
        
        "📊 **Your stats:**\n"
        f"• Max requests: {CFG.max_requests}\n"
        f"• Request expires in: {CFG.request_expire_hours // 24} days\n\n"
        
        "⚙️ **Commands:**\n"
        "/start - Start the bot\n"
        "/help - This help message\n"
        "/requests - View your requests\n"
        "/stats - View bot statistics\n"
        "/id - Get chat ID (group only)\n\n"
        
        "🔧 **Need help?** Contact @admin"
    )
    
    await message.reply_text(help_text, disable_web_page_preview=True)

@bot.on_message(filters.command("requests") & filters.private)
async def requests_command(client: Client, message: Message):
    """Handle /requests command"""
    user_id = message.from_user.id
    
    # Get user's requests
    user_requests = RequestManager.get_user_requests(user_id)
    
    if not user_requests:
        await message.reply_text(
            "📭 **No pending requests.**\n\n"
            "You haven't made any requests yet.\n"
            "Search for content and use the 'Request' button when it's not available."
        )
        return
    
    # Build requests list
    lines = ["📋 **Your Pending Requests:**\n"]
    
    for i, req in enumerate(user_requests, 1):
        title = req.get("title", "Unknown")
        year = req.get("year", "")
        media_type = req.get("media_type", "movie")
        created = req.get("created_at", datetime.now(timezone.utc))
        
        # Format date
        days_ago = (datetime.now(timezone.utc) - created).days
        
        icon = "🎬" if media_type == "movie" else "📺"
        lines.append(f"{i}. {icon} **{title}** {f'({year})' if year else ''}")
        lines.append(f"   ⏰ Requested {days_ago} day{'s' if days_ago != 1 else ''} ago")
        lines.append("")
    
    lines.append(f"📊 **Total:** {len(user_requests)}/{CFG.max_requests}")
    lines.append("\nℹ️ Use the buttons below to manage requests.")
    
    # Send message with keyboard
    await message.reply_text(
        "\n".join(lines),
        reply_markup=KeyboardBuilder.request_management_keyboard(user_requests),
        disable_web_page_preview=True
    )

@bot.on_message(filters.command("stats") & filters.private)
async def stats_command(client: Client, message: Message):
    """Handle /stats command"""
    try:
        # Get bot statistics
        total_users = users_col.count_documents({}) if users_col else 0
        total_files = files_col.count_documents({}) if files_col else 0
        total_requests = requests_col.count_documents({}) if requests_col else 0
        pending_requests = requests_col.count_documents({"status": "pending"}) if requests_col else 0
        
        # Get user's personal stats
        user_id = message.from_user.id
        user_data = users_col.find_one({"user_id": user_id}) if users_col else {}
        
        user_requests = user_data.get("requests_count", 0) if user_data else 0
        user_messages = user_data.get("messages_count", 0) if user_data else 0
        
        # Build stats message
        stats_text = (
            "📊 **Bot Statistics**\n\n"
            
            "👥 **Users:**\n"
            f"• Total users: {total_users:,}\n"
            f"• Your requests: {user_requests}\n"
            f"• Your messages: {user_messages}\n\n"
            
            "🎬 **Database:**\n"
            f"• Total files: {total_files:,}\n"
            f"• Total requests: {total_requests:,}\n"
            f"• Pending requests: {pending_requests:,}\n\n"
            
            "⚡ **Performance:**\n"
            f"• Uptime: {get_uptime()}\n"
            f"• Cache hits: N/A\n"
            f"• Memory usage: {get_memory_usage().get('percent', 0):.1f}%\n\n"
            
            "🔧 **Configuration:**\n"
            f"• Max requests/user: {CFG.max_requests}\n"
            f"• Request expiry: {CFG.request_expire_hours // 24} days\n"
            f"• Auto-notify: {'✅ On' if CFG.enable_auto_notify else '❌ Off'}"
        )
        
        await message.reply_text(stats_text, disable_web_page_preview=True)
        
    except Exception as e:
        log.error(f"Error in stats command: {e}")
        await message.reply_text("❌ Error fetching statistics.")

@bot.on_message(filters.command("id") & filters.group)
async def id_command(client: Client, message: Message):
    """Handle /id command in groups"""
    chat_id = message.chat.id
    chat_title = message.chat.title or "Unknown"
    
    response = (
        f"📋 **Chat Information**\n\n"
        f"• **Title:** {chat_title}\n"
        f"• **ID:** `{chat_id}`\n"
        f"• **Type:** {message.chat.type}\n"
        f"• **Members:** {message.chat.members_count if hasattr(message.chat, 'members_count') else 'Unknown'}\n\n"
        f"💡 **Note:** Use this ID in your .env file as ALLOWED_GROUP_ID"
    )
    
    await message.reply_text(response, quote=True)

@bot.on_message(filters.command("broadcast") & filters.private)
async def broadcast_command(client: Client, message: Message):
    """Handle broadcast command (admin only)"""
    # Check if user is admin (you can implement your own admin check)
    user_id = message.from_user.id
    if user_id != 123456789:  # Replace with your admin ID
        await message.reply_text("❌ Admin only command.")
        return
    
    # Check for message text
    if len(message.command) < 2:
        await message.reply_text("Usage: /broadcast <message>")
        return
    
    # Get broadcast message
    broadcast_text = " ".join(message.command[1:])
    
    # Send to all users
    sent = 0
    failed = 0
    
    await message.reply_text(f"📢 Starting broadcast to {users_col.count_documents({}):,} users...")
    
    for user in users_col.find({}, {"user_id": 1}):
        try:
            await client.send_message(user["user_id"], broadcast_text)
            sent += 1
            
            # Rate limiting
            if sent % 10 == 0:
                await asyncio.sleep(1)
                
        except Exception as e:
            failed += 1
            log.error(f"Failed to send to {user['user_id']}: {e}")
    
    await message.reply_text(
        f"✅ Broadcast completed!\n\n"
        f"• ✅ Sent: {sent}\n"
        f"• ❌ Failed: {failed}"
    )

# ============================================================
# GROUP SEARCH HANDLER
# ============================================================
@bot.on_message(filters.text & ~filters.private & ~filters.command)
async def group_search_handler(client: Client, message: Message):
    """Handle text messages in groups for searching"""
    
    # Check if in allowed group
    if message.chat.id != CFG.allowed_group_id:
        return
    
    # Check for maintenance mode
    if CFG.maintenance_mode:
        await message.reply_text("🔧 Bot is under maintenance. Please try again later.")
        return
    
    # Get search query
    query = message.text.strip()
    
    # Ignore very short queries
    if len(query) < 2:
        return
    
    # Ignore if looks like command
    if query.startswith("/"):
        return
    
    log.info(f"Search in group {message.chat.id}: '{query}'")
    
    try:
        # Search TMDB
        results = tmdb_client.search_multi(query)
        
        if not results:
            await message.reply_text(
                "❌ No results found.\n\n"
                "Try:\n"
                "• Different spelling\n"
                "• English title\n"
                "• Year (e.g., Avengers 2012)"
            )
            return
        
        # Send results
        await message.reply_text(
            f"🔍 **Found {len(results)} results for:** `{query}`\n"
            "Select the correct title:",
            reply_markup=KeyboardBuilder.search_results_keyboard(results)
        )
        
    except Exception as e:
        log.error(f"Search error: {e}")
        await message.reply_text("❌ Error searching. Please try again.")

# ============================================================
# CALLBACK QUERY HANDLER
# ============================================================
@bot.on_callback_query()
async def callback_query_handler(client: Client, callback_query: CallbackQuery):
    """Handle all callback queries"""
    
    user_id = callback_query.from_user.id
    data = callback_query.data
    
    # Register user
    await register_user(user_id, callback_query.from_user.username, callback_query.from_user.first_name)
    
    # Handle different callback types
    if data.startswith("detail:"):
        await handle_detail_callback(client, callback_query)
    
    elif data.startswith("request:"):
        await handle_request_callback(client, callback_query)
    
    elif data.startswith("cancel_request:"):
        await handle_cancel_request_callback(client, callback_query)
    
    elif data.startswith("admin_fill:"):
        await handle_admin_fill_callback(client, callback_query)
    
    elif data.startswith("admin_cancel:"):
        await handle_admin_cancel_callback(client, callback_query)
    
    elif data == "close":
        await callback_query.message.delete()
    
    elif data == "noop":
        await callback_query.answer()
    
    else:
        await callback_query.answer("Unknown action", show_alert=True)

async def handle_detail_callback(client: Client, callback_query: CallbackQuery):
    """Handle detail view callback"""
    try:
        # Parse callback data
        _, media_type, tmdb_id_str, _ = callback_query.data.split(":", 3)
        tmdb_id = int(tmdb_id_str)
        
        # Get media details
        media_data = tmdb_client.get_details(media_type, tmdb_id)
        if not media_data:
            await callback_query.answer("❌ Error fetching details", show_alert=True)
            return
        
        # Build detail card
        detail_text, poster_url = MessageBuilder.build_detail_card(media_data, media_type)
        
        # Get title and year for file matching
        title = media_data.get("title") if media_type == "movie" else media_data.get("name")
        date = media_data.get("release_date") if media_type == "movie" else media_data.get("first_air_date")
        year = date[:4] if date else None
        
        # Check availability
        matches = []
        is_available = False
        
        if CFG.enable_file_search:
            matches = find_matching_files(title, year)
            is_available = len(matches) > 0
        
        # Build availability message
        availability_text = MessageBuilder.build_availability_message(title, year, matches, is_available)
        
        # Combine messages
        full_text = f"{detail_text}\n\n{availability_text}"
        
        # Create keyboard
        keyboard = KeyboardBuilder.detail_keyboard(media_type, tmdb_id)
        
        # Send to user's PM
        try:
            if poster_url:
                await client.send_photo(
                    chat_id=callback_query.from_user.id,
                    photo=poster_url,
                    caption=full_text,
                    reply_markup=keyboard,
                    parse_mode="markdown"
                )
            else:
                await client.send_message(
                    chat_id=callback_query.from_user.id,
                    text=full_text,
                    reply_markup=keyboard,
                    parse_mode="markdown",
                    disable_web_page_preview=True
                )
            
            await callback_query.answer("📨 Sent to your PM!", show_alert=False)
            
        except Exception as e:
            log.error(f"Error sending to PM: {e}")
            await callback_query.answer(
                "❌ Cannot send to PM. Please start the bot first!",
                show_alert=True
            )
            
            # Send help message in group
            username = await get_bot_username_safe()
            help_msg = await callback_query.message.reply_text(
                "⚠️ **Cannot send to your PM.**\n\n"
                "Please:\n"
                "1. Click the button below\n"
                "2. Press 'Start' in the bot\n"
                "3. Try again\n\n"
                "If issues persist, contact @admin",
                reply_markup=KeyboardBuilder.start_pm_keyboard()
            )
            
            # Delete help message after 30 seconds
            await asyncio.sleep(30)
            try:
                await help_msg.delete()
            except:
                pass
    
    except Exception as e:
        log.error(f"Error in detail callback: {e}")
        await callback_query.answer("❌ Error processing request", show_alert=True)

async def handle_request_callback(client: Client, callback_query: CallbackQuery):
    """Handle request callback"""
    try:
        # Parse callback data
        _, media_type, tmdb_id_str = callback_query.data.split(":", 2)
        tmdb_id = int(tmdb_id_str)
        user_id = callback_query.from_user.id
        
        # Get media details
        media_data = tmdb_client.get_details(media_type, tmdb_id)
        if not media_data:
            await callback_query.answer("❌ Error fetching details", show_alert=True)
            return
        
        # Get title and year
        title = media_data.get("title") if media_type == "movie" else media_data.get("name")
        date = media_data.get("release_date") if media_type == "movie" else media_data.get("first_air_date")
        year = date[:4] if date else None
        
        # Check if already available (prevent unnecessary requests)
        if CFG.enable_file_search:
            matches = find_matching_files(title, year, limit=1)
            if matches:
                await callback_query.answer(
                    "✅ Already available! Check details in PM.",
                    show_alert=True
                )
                return
        
        # Check request limit
        current_requests = RequestManager.get_user_requests_count(user_id)
        if current_requests >= CFG.max_requests:
            # Get user's requests for management
            user_requests = RequestManager.get_user_requests(user_id)
            
            await callback_query.answer("❌ Request limit reached!", show_alert=True)
            
            # Send management message
            await client.send_message(
                user_id,
                f"⚠️ **Request Limit Reached**\n\n"
                f"You have {current_requests}/{CFG.max_requests} pending requests.\n"
                f"Please cancel some requests before making new ones:",
                reply_markup=KeyboardBuilder.request_management_keyboard(user_requests)
            )
            return
        
        # Create request
        success = RequestManager.create_request(user_id, media_type, tmdb_id, title, year)
        
        if not success:
            await callback_query.answer("❌ Error creating request", show_alert=True)
            return
        
        # Notify admin channel
        if CFG.admin_req_channel_id:
            try:
                tmdb_link = f"https://www.themoviedb.org/{media_type}/{tmdb_id}"
                user_info = f"@{callback_query.from_user.username}" if callback_query.from_user.username else f"User #{user_id}"
                
                admin_message = (
                    f"📥 **NEW REQUEST**\n\n"
                    f"👤 **User:** {user_info}\n"
                    f"🎬 **Type:** {media_type.upper()}\n"
                    f"📝 **Title:** {title} {f'({year})' if year else ''}\n"
                    f"🔗 **TMDB:** {tmdb_link}\n"
                    f"⏰ **Time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
                
                await client.send_message(
                    CFG.admin_req_channel_id,
                    admin_message,
                    reply_markup=KeyboardBuilder.admin_actions_keyboard(media_type, tmdb_id, user_id),
                    disable_web_page_preview=True
                )
                
            except Exception as e:
                log.error(f"Error sending to admin channel: {e}")
        
        # Notify user
        await callback_query.answer(
            "✅ Request sent successfully!\n"
            "You'll be notified when available.",
            show_alert=True
        )
        
        # Send confirmation to user
        await client.send_message(
            user_id,
            f"✅ **Request Submitted**\n\n"
            f"**Title:** {title} {f'({year})' if year else ''}\n"
            f"**Type:** {media_type.capitalize()}\n"
            f"**Status:** Pending\n\n"
            f"We'll notify you when this becomes available.\n"
            f"Requests expire after {CFG.request_expire_hours // 24} days."
        )
    
    except Exception as e:
        log.error(f"Error in request callback: {e}")
        await callback_query.answer("❌ Error processing request", show_alert=True)

async def handle_cancel_request_callback(client: Client, callback_query: CallbackQuery):
    """Handle cancel request callback"""
    try:
        # Parse callback data
        _, request_id = callback_query.data.split(":", 1)
        user_id = callback_query.from_user.id
        
        # Cancel request
        success = RequestManager.cancel_request(request_id, user_id)
        
        if success:
            await callback_query.answer("✅ Request cancelled", show_alert=True)
            await callback_query.message.delete()
            
            # Send confirmation
            await client.send_message(
                user_id,
                "✅ Request cancelled successfully."
            )
        else:
            await callback_query.answer("❌ Error cancelling request", show_alert=True)
    
    except Exception as e:
        log.error(f"Error in cancel request callback: {e}")
        await callback_query.answer("❌ Error processing request", show_alert=True)

async def handle_admin_fill_callback(client: Client, callback_query: CallbackQuery):
    """Handle admin fill callback"""
    # Check if admin (simple check - implement proper admin check)
    if callback_query.from_user.id != 123456789:  # Replace with your admin ID
        await callback_query.answer("❌ Admin only", show_alert=True)
        return
    
    try:
        # Parse callback data
        _, media_type, tmdb_id_str = callback_query.data.split(":", 2)
        tmdb_id = int(tmdb_id_str)
        
        # Mark requests as filled
        count = RequestManager.mark_requests_completed(media_type, tmdb_id)
        
        await callback_query.answer(
            f"✅ Marked {count} requests as filled",
            show_alert=True
        )
        
        # Update message
        await callback_query.message.edit_text(
            f"{callback_query.message.text}\n\n"
            f"✅ **Filled by admin**\n"
            f"• Requests completed: {count}\n"
            f"• Time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}",
            reply_markup=None
        )
    
    except Exception as e:
        log.error(f"Error in admin fill callback: {e}")
        await callback_query.answer("❌ Error processing request", show_alert=True)

async def handle_admin_cancel_callback(client: Client, callback_query: CallbackQuery):
    """Handle admin cancel callback"""
    # Check if admin
    if callback_query.from_user.id != 123456789:  # Replace with your admin ID
        await callback_query.answer("❌ Admin only", show_alert=True)
        return
    
    try:
        # Parse callback data
        _, media_type, tmdb_id_str, user_id_str = callback_query.data.split(":", 3)
        tmdb_id = int(tmdb_id_str)
        user_id = int(user_id_str)
        
        # Cancel user's requests for this media
        result = requests_col.update_many(
            {
                "user_id": user_id,
                "media_type": media_type,
                "tmdb_id": tmdb_id,
                "status": "pending"
            },
            {
                "$set": {
                    "status": "cancelled_by_admin",
                    "cancelled_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        count = result.modified_count
        
        await callback_query.answer(
            f"✅ Cancelled {count} requests for user",
            show_alert=True
        )
        
        # Update message
        await callback_query.message.edit_text(
            f"{callback_query.message.text}\n\n"
            f"❌ **Cancelled by admin**\n"
            f"• User: {user_id}\n"
            f"• Requests cancelled: {count}\n"
            f"• Time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}",
            reply_markup=None
        )
        
        # Notify user
        try:
            await client.send_message(
                user_id,
                f"⚠️ **Request Cancelled**\n\n"
                f"Your request was cancelled by admin.\n"
                f"Contact @admin for more information."
            )
        except:
            pass
    
    except Exception as e:
        log.error(f"Error in admin cancel callback: {e}")
        await callback_query.answer("❌ Error processing request", show_alert=True)

# ============================================================
# AUTO-NOTIFY POLLER
# ============================================================
class AutoNotifyPoller:
    """Poller for auto-notifying users when files become available"""
    
    def __init__(self):
        self.running = False
        self.last_processed_id = None
        self.notification_queue = []
        self.batch_size = 50
        
    def start(self):
        """Start the poller"""
        if self.running:
            return
        
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        log.info("Auto-notify poller started")
    
    def stop(self):
        """Stop the poller"""
        self.running = False
        log.info("Auto-notify poller stopped")
    
    def _run(self):
        """Main poller loop"""
        # Load last processed ID
        self._load_last_processed_id()
        
        while self.running and CFG.enable_auto_notify:
            try:
                self._process_new_files()
                self._process_notification_queue()
                
                # Cleanup expired requests periodically
                if time.time() % 3600 < CFG.poll_seconds:  # Once per hour
                    RequestManager.cleanup_expired_requests()
                
                time.sleep(CFG.poll_seconds)
                
            except Exception as e:
                log.error(f"Poller error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _load_last_processed_id(self):
        """Load last processed file ID from database"""
        try:
            if not mongo_ping():
                return
            
            meta = meta_col.find_one({"key": "last_processed_file_id"})
            if meta and meta.get("value"):
                self.last_processed_id = ObjectId(meta["value"])
        except Exception as e:
            log.error(f"Error loading last processed ID: {e}")
    
    def _save_last_processed_id(self, file_id: ObjectId):
        """Save last processed file ID to database"""
        try:
            if not mongo_ping():
                return
            
            meta_col.update_one(
                {"key": "last_processed_file_id"},
                {"$set": {"value": str(file_id), "updated_at": datetime.now(timezone.utc)}},
                upsert=True
            )
            self.last_processed_id = file_id
        except Exception as e:
            log.error(f"Error saving last processed ID: {e}")
    
    def _process_new_files(self):
        """Process new files from database"""
        try:
            if not mongo_ping():
                return
            
            # Build query
            query = {}
            if self.last_processed_id:
                query["_id"] = {"$gt": self.last_processed_id}
            
            # Get new files
            cursor = files_col.find(query, {"_id": 1, "file_name": 1}).sort("_id", 1).limit(self.batch_size)
            new_files = list(cursor)
            
            if not new_files:
                return
            
            # Get all pending requests
            pending_requests = list(requests_col.find(
                {"status": "pending"},
                {"user_id": 1, "media_type": 1, "tmdb_id": 1, "title": 1, "year": 1}
            ).limit(1000))
            
            if not pending_requests:
                # Just update last processed ID
                self._save_last_processed_id(new_files[-1]["_id"])
                return
            
            # Process each file
            for file_doc in new_files:
                filename = file_doc.get("file_name", "")
                file_id = file_doc["_id"]
                
                # Normalize filename
                norm_filename, tokens, file_year = normalize_filename(filename)
                
                if not tokens:
                    continue
                
                # Check against each pending request
                for req in pending_requests:
                    req_title = req.get("title", "")
                    req_year = req.get("year")
                    
                    # Check year match
                    if req_year and file_year and req_year != file_year:
                        continue
                    
                    # Calculate similarity
                    similarity = calculate_similarity(req_title, norm_filename)
                    
                    # Threshold for match
                    threshold = 0.7 if req_year else 0.8
                    
                    if similarity >= threshold:
                        # Found a match!
                        self.notification_queue.append({
                            "user_id": req["user_id"],
                            "request_id": req["_id"],
                            "media_type": req["media_type"],
                            "tmdb_id": req["tmdb_id"],
                            "title": req_title,
                            "year": req_year,
                            "filename": filename,
                            "similarity": similarity
                        })
                
                # Update last processed ID
                self._save_last_processed_id(file_id)
            
            log.info(f"Processed {len(new_files)} new files, found {len(self.notification_queue)} matches")
            
        except Exception as e:
            log.error(f"Error processing new files: {e}")
    
    def _process_notification_queue(self):
        """Process notification queue and send notifications"""
        if not self.notification_queue:
            return
        
        log.info(f"Processing {len(self.notification_queue)} notifications")
        
        for notification in self.notification_queue[:10]:  # Process 10 at a time
            try:
                self._send_notification(notification)
                
                # Mark request as done
                requests_col.update_one(
                    {"_id": notification["request_id"]},
                    {
                        "$set": {
                            "status": "done",
                            "done_at": datetime.now(timezone.utc),
                            "matched_file": notification["filename"],
                            "updated_at": datetime.now(timezone.utc)
                        }
                    }
                )
                
            except Exception as e:
                log.error(f"Error sending notification: {e}")
        
        # Clear processed notifications
        self.notification_queue = self.notification_queue[10:]
    
    def _send_notification(self, notification: Dict):
        """Send notification to user"""
        try:
            # Get bot instance
            bot_instance = get_bot()
            if not bot_instance or not bot_instance.is_connected:
                return
            
            user_id = notification["user_id"]
            title = notification["title"]
            year = notification["year"]
            filename = notification["filename"]
            media_type = notification["media_type"]
            tmdb_id = notification["tmdb_id"]
            
            # Truncate filename if too long
            if len(filename) > 50:
                display_filename = filename[:47] + "..."
            else:
                display_filename = filename
            
            # Build notification message
            message = (
                f"🎉 **Good News!**\n\n"
                f"Your requested content is now available!\n\n"
                f"**Title:** {title} {f'({year})' if year else ''}\n"
                f"**File:** `{display_filename}`\n\n"
                f"👉 Search in the group to get download links.\n"
                f"👉 Group එකේ search කරලා download links ගන්න."
            )
            
            # Send message
            asyncio.run_coroutine_threadsafe(
                bot_instance.send_message(user_id, message),
                bot_instance.loop
            )
            
            log.info(f"Sent notification to user {user_id} for {title}")
            
        except Exception as e:
            log.error(f"Error in send_notification: {e}")

# Global poller instance
poller = AutoNotifyPoller()

# ============================================================
# ERROR HANDLING AND SIGNAL HANDLERS
# ============================================================
def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    log.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def signal_handler(signum, frame):
    """Handle termination signals"""
    log.info(f"Received signal {signum}, shutting down...")
    
    # Stop poller
    poller.stop()
    
    # Disconnect MongoDB
    mongo_manager.disconnect()
    
    log.info("Shutdown complete")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================
# MAIN FUNCTION
# ============================================================
async def main_async():
    """Main async function"""
    
    # Validate configuration
    config_errors = CFG.validate()
    if config_errors:
        log.critical("Configuration errors:")
        for error in config_errors:
            log.critical(f"  - {error}")
        log.critical("Please check your environment variables and restart.")
        return
    
    log.info("=" * 50)
    log.info("Ultra Pro Max Bot Starting...")
    log.info("=" * 50)
    
    # Print configuration (mask sensitive data)
    masked_mongo = CFG.mongo_uri
    if "@" in masked_mongo:
        parts = masked_mongo.split("@")
        masked_mongo = f"mongodb://****:****@{parts[1]}"
    
    log.info(f"Bot: @{CFG.bot_token.split(':')[0]}")
    log.info(f"Group ID: {CFG.allowed_group_id}")
    log.info(f"Admin Channel: {CFG.admin_req_channel_id}")
    log.info(f"MongoDB: {masked_mongo}")
    log.info(f"Debug Mode: {CFG.debug_mode}")
    log.info(f"Features - Search: {CFG.enable_file_search}, "
             f"Requests: {CFG.enable_request_system}, "
             f"Auto-notify: {CFG.enable_auto_notify}")
    log.info("=" * 50)
    
    # Initialize MongoDB
    log.info("Connecting to MongoDB...")
    if not init_mongo():
        log.critical("Failed to connect to MongoDB. Exiting.")
        return
    
    # Start Flask web server in background thread
    log.info(f"Starting web server on port {CFG.port}...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Initialize bot
    global bot
    bot = Client(
        name=CFG.app_name,
        api_id=CFG.api_id,
        api_hash=CFG.api_hash,
        bot_token=CFG.bot_token,
        workers=10,
        sleep_threshold=30,
        parse_mode="markdown"
    )
    
    # Register handlers
    log.info("Registering bot handlers...")
    
    # Start bot
    log.info("Starting bot...")
    await bot.start()
    
    # Get bot info
    me = await bot.get_me()
    global bot_username
    bot_username = me.username
    log.info(f"Bot started: @{bot_username}")
    
    # Send startup notification to admin channel
    if CFG.admin_req_channel_id and CFG.log_channel_id:
        try:
            startup_msg = (
                f"🤖 **Bot Started Successfully**\n\n"
                f"• **Name:** @{bot_username}\n"
                f"• **Time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"• **Uptime:** {get_uptime()}\n"
                f"• **Version:** 2.0.0\n\n"
                f"✅ All systems operational."
            )
            
            await bot.send_message(CFG.admin_req_channel_id, startup_msg)
            
        except Exception as e:
            log.error(f"Failed to send startup notification: {e}")
    
    # Start auto-notify poller
    if CFG.enable_auto_notify:
        log.info("Starting auto-notify poller...")
        poller.start()
    
    # Bot is ready
    log.info("Bot is ready and running!")
    log.info(f"Group: {CFG.allowed_group_id}")
    log.info(f"Admin Channel: {CFG.admin_req_channel_id}")
    log.info("=" * 50)
    
    # Keep bot running
    await idle()
    
    # Stop bot
    log.info("Stopping bot...")
    await bot.stop()
    
    # Stop poller
    poller.stop()
    
    # Disconnect MongoDB
    mongo_manager.disconnect()
    
    log.info("Bot stopped successfully")

def main():
    """Main entry point"""
    try:
        # Import asyncio
        import asyncio
        
        # Run main async function
        asyncio.run(main_async())
        
    except KeyboardInterrupt:
        log.info("Bot stopped by user")
    except Exception as e:
        log.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

# ============================================================
# RENDER DEPLOYMENT COMPATIBILITY
# ============================================================
if __name__ == "__main__":
    # Check if we're running on Render
    is_render = os.getenv("RENDER", "").lower() == "true"
    
    if is_render:
        log.info("Running on Render platform")
        
        # Render-specific optimizations
        CFG.connection_pool_size = 5  # Reduce for free tier
        CFG.scan_limit = 1000  # Reduce scan limit
        CFG.poll_seconds = 60  # Increase poll interval
        
        # Ensure web server runs
        web_thread = threading.Thread(target=run_flask, daemon=True)
        web_thread.start()
        
        # Import asyncio for Render
        import asyncio
        
        # Run bot in main thread
        try:
            asyncio.run(main_async())
        except KeyboardInterrupt:
            log.info("Bot stopped")
        except Exception as e:
            log.critical(f"Bot crashed: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Local development
        main()
