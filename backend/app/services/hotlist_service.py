"""Hotlist service with O(1) Redis-backed cache, deduplication, and cooldown enforcement."""

import asyncio
import logging
import uuid
from datetime import UTC, datetime

import redis.asyncio as redis
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.hotlist import HotListAlert, HotListEntry
from app.schemas.hotlist import HotListAlertUpdate, HotListEntryCreate, HotListEntryUpdate
from app.services.redis_client import get_redis

logger = logging.getLogger(__name__)

# ─── Redis-backed O(1) Hotlist Cache ───

CACHE_KEY = "reposcan:hotlist:active_plates"
COOLDOWN_PREFIX = "reposcan:hotlist:cooldown:"
DEFAULT_COOLDOWN_SECONDS = 300  # 5-minute cooldown between alerts for same entry


class HotlistCache:
    """Maintains a Redis SET of active plate texts for O(1) membership checks.

    On startup, loads all active plates from the DB into a Redis set.
    Mutations (create/update/delete) keep the cache in sync.
    check_plate() does a single SISMEMBER call — O(1).
    """

    def __init__(self) -> None:
        self._redis: redis.Redis | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self._redis = await get_redis()

    async def disconnect(self) -> None:
        # Connection lifecycle managed by redis_client module
        self._redis = None

    async def load_from_db(self, db: AsyncSession) -> None:
        """Bulk-load all active, non-expired plate texts into the Redis set."""
        now = datetime.now(UTC)
        result = await db.execute(
            select(HotListEntry.plate_text).where(
                and_(
                    HotListEntry.is_active.is_(True),
                    (HotListEntry.expires_at.is_(None)) | (HotListEntry.expires_at > now),
                )
            )
        )
        plates = [row[0] for row in result.all()]

        if self._redis:
            async with self._lock:
                pipe = self._redis.pipeline()
                pipe.delete(CACHE_KEY)
                if plates:
                    pipe.sadd(CACHE_KEY, *plates)
                await pipe.execute()
            logger.info("Hotlist cache loaded: %d active plates", len(plates))

    async def is_plate_cached(self, plate_text: str) -> bool:
        """O(1) membership check via Redis SISMEMBER."""
        if not self._redis:
            return False
        return bool(await self._redis.sismember(CACHE_KEY, plate_text.upper().strip()))

    async def add_plate(self, plate_text: str) -> None:
        if self._redis:
            await self._redis.sadd(CACHE_KEY, plate_text.upper().strip())

    async def remove_plate(self, plate_text: str) -> None:
        if self._redis:
            await self._redis.srem(CACHE_KEY, plate_text.upper().strip())

    # ─── Cooldown enforcement ───

    async def check_cooldown(self, hotlist_entry_id: uuid.UUID) -> bool:
        """Return True if the cooldown is active (alert should be suppressed)."""
        if not self._redis:
            return False
        key = f"{COOLDOWN_PREFIX}{hotlist_entry_id}"
        return bool(await self._redis.exists(key))

    async def set_cooldown(
        self, hotlist_entry_id: uuid.UUID, seconds: int = DEFAULT_COOLDOWN_SECONDS
    ) -> None:
        """Set a cooldown TTL key to suppress duplicate alerts."""
        if self._redis:
            key = f"{COOLDOWN_PREFIX}{hotlist_entry_id}"
            await self._redis.set(key, "1", ex=seconds)

    # ─── Deduplication lock ───

    async def acquire_alert_lock(
        self, hotlist_entry_id: uuid.UUID, detection_id: uuid.UUID
    ) -> bool:
        """Atomic lock to prevent duplicate alert creation under concurrent requests.

        Returns True if lock was acquired (caller should proceed), False if another
        request already holds it (caller should skip).
        """
        if not self._redis:
            return True
        key = f"reposcan:hotlist:alert_lock:{hotlist_entry_id}:{detection_id}"
        # SET NX with 30s TTL — if the key already exists, another request won.
        acquired = await self._redis.set(key, "1", nx=True, ex=30)
        return bool(acquired)


# Module-level singleton
hotlist_cache = HotlistCache()


# ─── Entry CRUD (cache-aware) ───


async def create_entry(db: AsyncSession, data: HotListEntryCreate) -> HotListEntry:
    plate = data.plate_text.upper().strip()

    # Deduplicate: if an active entry with same plate already exists, return it
    existing = await db.execute(
        select(HotListEntry).where(
            and_(
                HotListEntry.plate_text == plate,
                HotListEntry.is_active.is_(True),
            )
        )
    )
    if (found := existing.scalar_one_or_none()) is not None:
        return found

    entry = HotListEntry(
        plate_text=plate,
        plate_state=data.plate_state,
        case_number=data.case_number,
        lender_name=data.lender_name,
        lender_contact=data.lender_contact,
        order_type=data.order_type,
        vehicle_year=data.vehicle_year,
        vehicle_make=data.vehicle_make,
        vehicle_model=data.vehicle_model,
        vehicle_color=data.vehicle_color,
        vin=data.vin,
        debtor_name=data.debtor_name,
        debtor_address=data.debtor_address,
        debtor_phone=data.debtor_phone,
        priority=data.priority,
        notes=data.notes,
        expires_at=data.expires_at,
        extra=data.extra,
    )
    db.add(entry)
    await db.flush()

    # Sync cache
    await hotlist_cache.add_plate(plate)
    return entry


async def update_entry(
    db: AsyncSession, entry_id: uuid.UUID, data: HotListEntryUpdate
) -> HotListEntry | None:
    entry = await db.get(HotListEntry, entry_id)
    if not entry:
        return None

    old_plate = entry.plate_text
    old_active = entry.is_active
    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == "plate_text" and value:
            value = value.upper().strip()
        setattr(entry, field, value)

    await db.flush()

    # Sync cache when plate text or active status changes
    new_plate = entry.plate_text
    new_active = entry.is_active
    if old_plate != new_plate or old_active != new_active:
        if old_active:
            await hotlist_cache.remove_plate(old_plate)
        if new_active:
            await hotlist_cache.add_plate(new_plate)

    return entry


async def get_entry(db: AsyncSession, entry_id: uuid.UUID) -> HotListEntry | None:
    # No eager load of alerts — the response schema doesn't include them,
    # and an entry with thousands of historical alerts would be expensive.
    return await db.get(HotListEntry, entry_id)


async def list_entries(
    db: AsyncSession,
    active_only: bool = True,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[HotListEntry], int]:
    query = select(HotListEntry)
    count_query = select(func.count(HotListEntry.id))

    if active_only:
        query = query.where(HotListEntry.is_active.is_(True))
        count_query = count_query.where(HotListEntry.is_active.is_(True))

    total = (await db.execute(count_query)).scalar() or 0

    query = (
        query.order_by(HotListEntry.priority.desc(), HotListEntry.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    result = await db.execute(query)
    return list(result.scalars().all()), total


async def delete_entry(db: AsyncSession, entry_id: uuid.UUID) -> bool:
    entry = await db.get(HotListEntry, entry_id)
    if not entry:
        return False

    plate = entry.plate_text
    await db.delete(entry)
    try:
        await db.flush()
    except Exception:
        logger.error("Failed to flush hotlist entry delete for %s", entry_id)
        raise
    # Remove from cache only after successful flush
    await hotlist_cache.remove_plate(plate)
    return True


# ─── Plate Matching (O(1) fast path + DB verification) ───


async def check_plate_against_hotlist(db: AsyncSession, plate_text: str) -> list[HotListEntry]:
    """Two-tier lookup: O(1) Redis cache check, then DB verification only on hit."""
    plate_upper = plate_text.upper().strip()

    # Fast path: O(1) Redis SET membership check
    if not await hotlist_cache.is_plate_cached(plate_upper):
        return []

    # Cache hit — verify against DB (handles expiration edge cases)
    now = datetime.now(UTC)
    result = await db.execute(
        select(HotListEntry).where(
            and_(
                HotListEntry.is_active.is_(True),
                HotListEntry.plate_text == plate_upper,
                (HotListEntry.expires_at.is_(None)) | (HotListEntry.expires_at > now),
            )
        )
    )
    entries = list(result.scalars().all())

    # If DB says no match, cache is stale — clean it up
    if not entries:
        await hotlist_cache.remove_plate(plate_upper)

    return entries


# ─── Alert Creation with Deduplication + Cooldown ───


async def create_alert(
    db: AsyncSession,
    hotlist_entry_id: uuid.UUID,
    detection_id: uuid.UUID | None,
    agent_id: uuid.UUID | None,
    plate_text: str,
    confidence: float,
    latitude: float | None = None,
    longitude: float | None = None,
    address: str | None = None,
) -> HotListAlert | None:
    """Create an alert with cooldown enforcement and deduplication.

    Returns None if the alert is suppressed by cooldown or lost a dedup race.
    """
    # 1. Cooldown check — suppress rapid re-alerts for the same hotlist entry
    if await hotlist_cache.check_cooldown(hotlist_entry_id):
        logger.debug(
            "Alert suppressed by cooldown: entry=%s plate=%s",
            hotlist_entry_id,
            plate_text,
        )
        return None

    # 2. Deduplication lock — prevent concurrent requests creating duplicate alerts
    if detection_id and not await hotlist_cache.acquire_alert_lock(hotlist_entry_id, detection_id):
        logger.debug(
            "Alert suppressed by dedup lock: entry=%s detection=%s",
            hotlist_entry_id,
            detection_id,
        )
        return None

    alert = HotListAlert(
        hotlist_entry_id=hotlist_entry_id,
        detection_id=detection_id,
        agent_id=agent_id,
        plate_text_matched=plate_text,
        match_confidence=confidence,
        latitude=latitude,
        longitude=longitude,
        address=address,
    )
    db.add(alert)
    await db.flush()

    # 3. Set cooldown so the same entry won't fire again within the window
    await hotlist_cache.set_cooldown(hotlist_entry_id)

    return alert


# ─── Alert Queries ───


async def list_alerts(
    db: AsyncSession,
    status: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[HotListAlert], int]:
    query = select(HotListAlert).options(selectinload(HotListAlert.hotlist_entry))
    count_query = select(func.count(HotListAlert.id))

    if status:
        query = query.where(HotListAlert.status == status)
        count_query = count_query.where(HotListAlert.status == status)

    total = (await db.execute(count_query)).scalar() or 0

    query = (
        query.order_by(HotListAlert.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    result = await db.execute(query)
    return list(result.scalars().all()), total


async def update_alert(
    db: AsyncSession, alert_id: uuid.UUID, data: HotListAlertUpdate
) -> HotListAlert | None:
    alert = await db.get(HotListAlert, alert_id)
    if not alert:
        return None

    if data.status:
        alert.status = data.status
        if data.status == "acknowledged":
            alert.acknowledged_at = datetime.now(UTC)
        elif data.status in ("resolved", "false_positive"):
            alert.resolved_at = datetime.now(UTC)
    if data.notes is not None:
        alert.notes = data.notes

    await db.flush()
    return alert
