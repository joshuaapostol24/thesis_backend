import os
import httpx
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient

router = APIRouter(prefix="/news", tags=["News"])

_mongo_client    = None
_news_collection = None

def _get_collection():
    global _mongo_client, _news_collection
    if _news_collection is None:
        mongo_url        = os.environ.get("MONGODB_URL", "mongodb://localhost:27017")
        _mongo_client    = MongoClient(mongo_url)
        db               = _mongo_client[os.environ.get("MONGODB_DB", "resq")]
        _news_collection = db["news"]
    return _news_collection


class NewsCreate(BaseModel):
    title:    str
    category: Optional[str] = "General News"
    priority: Optional[str] = "Low"
    date:     Optional[str] = None
    audience: Optional[str] = "All Residents"
    pinned:   Optional[str] = "No"
    message:  str


class NewsEdit(BaseModel):
    title:    Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    date:     Optional[str] = None
    audience: Optional[str] = None
    pinned:   Optional[str] = None
    message:  Optional[str] = None


def notify_users_of_news(news_item: dict):
    try:
        service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        httpx.post(
            "https://jpovamcznyzoemcnjrgs.supabase.co/functions/v1/send-news-push",
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {service_role_key}",
            },
            json={
                "title":    news_item.get("title"),
                "message":  news_item.get("message"),
                "category": news_item.get("category"),
                "priority": news_item.get("priority"),
                "date":     news_item.get("date"),
                "audience": news_item.get("audience"),
            },
            timeout=10.0
        )
    except Exception as e:
        print("Notification failed:", str(e))


def _handle_pinning(collection, new_item_is_pinned: bool):
    if not new_item_is_pinned:
        return
    collection.update_many({"pinned": "Yes"}, {"$set": {"pinned": "No"}})


@router.get("/all")
def get_all_news():
    collection = _get_collection()
    news = list(collection.find({}))
    for item in news:
        item["id"] = str(item["_id"])
        del item["_id"]
    return news


@router.get("/public")
def get_public_news():
    collection = _get_collection()
    return list(collection.find({}, {"_id": 0}))


@router.post("/create")
def create_news(data: NewsCreate):
    collection = _get_collection()
    news_item  = data.dict()
    news_item["createdAt"] = datetime.now(timezone.utc).isoformat()
    is_pinned  = news_item.get("pinned", "No") == "Yes"
    _handle_pinning(collection, is_pinned)
    collection.insert_one(news_item)
    notify_users_of_news(news_item)
    return {
        "success": True,
        "message": "Announcement published successfully",
        "pinned":  is_pinned,
        "note": (
            "Previous pinned announcement moved to Latest."
            if is_pinned else
            "Announcement added to Latest."
        ),
    }


@router.patch("/edit/{id}")
def edit_news(id: str, data: NewsEdit):
    """
    Edit an existing announcement by ID.
    Only provided fields are updated — omitted fields are left unchanged.
    If pinned is changed to 'Yes', the previous pinned post is demoted.
    """
    from bson import ObjectId
    collection = _get_collection()

    try:
        oid = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid news ID")

    existing = collection.find_one({"_id": oid})
    if not existing:
        raise HTTPException(status_code=404, detail="News item not found")

    # Only update fields that were actually provided
    updates = {k: v for k, v in data.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    # Add edited timestamp
    updates["editedAt"] = datetime.now(timezone.utc).isoformat()

    # Handle pinning if pinned field is being changed to Yes
    if updates.get("pinned") == "Yes" and existing.get("pinned") != "Yes":
        _handle_pinning(collection, new_item_is_pinned=True)

    collection.update_one({"_id": oid}, {"$set": updates})

    return {
        "success":       True,
        "message":       "Announcement updated successfully",
        "updated_fields": list(updates.keys()),
    }


@router.delete("/delete/{id}")
def delete_news(id: str):
    from bson import ObjectId
    collection = _get_collection()
    result = collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 0:
        return {"success": False, "message": "News not found"}
    return {"success": True, "message": "News deleted successfully"}


@router.patch("/pin/{id}")
def pin_news(id: str):
    from bson import ObjectId
    collection = _get_collection()
    try:
        oid = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid news ID")
    target = collection.find_one({"_id": oid})
    if not target:
        raise HTTPException(status_code=404, detail="News item not found")
    _handle_pinning(collection, new_item_is_pinned=True)
    collection.update_one({"_id": oid}, {"$set": {"pinned": "Yes"}})
    return {
        "success": True,
        "message": f"'{target['title']}' is now pinned. Previous pin moved to Latest.",
    }


@router.patch("/unpin/{id}")
def unpin_news(id: str):
    from bson import ObjectId
    collection = _get_collection()
    try:
        oid = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid news ID")
    result = collection.update_one({"_id": oid}, {"$set": {"pinned": "No"}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="News item not found")
    return {"success": True, "message": "Announcement unpinned and moved to Latest."}