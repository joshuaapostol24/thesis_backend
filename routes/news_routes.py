import os
import httpx
from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient

# =========================================================
# ROUTER
# =========================================================
router = APIRouter(prefix="/news", tags=["News"])

# =========================================================
# MONGODB CLIENT
# =========================================================
_mongo_client = None
_news_collection = None

def _get_collection():
    global _mongo_client, _news_collection
    if _news_collection is None:
        mongo_url = os.environ.get("MONGODB_URL", "mongodb://localhost:27017")
        _mongo_client = MongoClient(mongo_url)
        db = _mongo_client[os.environ.get("MONGODB_DB", "resq")]
        _news_collection = db["news"]
    return _news_collection

# =========================================================
# MODELS
# =========================================================
class NewsCreate(BaseModel):
    title: str
    category: Optional[str] = "General News"
    priority: Optional[str] = "Low"
    date: Optional[str] = None
    audience: Optional[str] = "All Residents"
    pinned: Optional[str] = "No"
    message: str

# =========================================================
# NOTIFY USERS
# =========================================================
def notify_users_of_news(news_item: dict):
    try:
        service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        
        response = httpx.post(
            "https://jpovamcznyzoemcnjrgs.supabase.co/functions/v1/send-news-push",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {service_role_key}",
            },
            json={
                "title": news_item.get("title"),
                "message": news_item.get("message"),
                "category": news_item.get("category"),
                "priority": news_item.get("priority"),
            },
            timeout=10.0
        )
        
        print("Notification result:", response.status_code, response.text)

    except Exception as e:
        print("Notification failed:", str(e))

# =========================================================
# ROUTES
# =========================================================
@router.get("/all")
def get_all_news():
    collection = _get_collection()
    news = list(collection.find({}, {"_id": 0}))
    return news

@router.get("/public")
def get_public_news():
    collection = _get_collection()
    news = list(collection.find({}, {"_id": 0}))
    return news

@router.post("/create")
def create_news(data: NewsCreate):
    collection = _get_collection()
    news_item = data.dict()
    news_item["createdAt"] = datetime.now(timezone.utc).isoformat()
    collection.insert_one(news_item)

    # Send push notification
    notify_users_of_news(news_item)

    return {"success": True, "message": "Announcement published successfully"}