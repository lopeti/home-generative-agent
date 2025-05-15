"""Home Generative Agent Initialization."""
from __future__ import annotations

# vendor override for stdlib imghdr
import sys
from . import imghdr as _imghdr
sys.modules['imghdr'] = _imghdr

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import homeassistant.util.dt as dt_util
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_API_KEY,
    EVENT_STATE_CHANGED,
    Platform,
)
from homeassistant.core import HomeAssistant, callback, Event
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.httpx_client import get_async_client
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import ConfigurableField
from langchain_ollama import ChatOllama, OllamaEmbeddings
# Monkey-patch to strip out unsupported 'proxies' parameter
from langchain_openai import ChatOpenAI

_orig_validate_env = ChatOpenAI.validate_environment

def _patched_validate_environment(self, client_params, sync_specific):
    client_params.pop("proxies", None)
    return _orig_validate_env(self, client_params, sync_specific)

ChatOpenAI.validate_environment = _patched_validate_environment
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from .const import (
    CONF_SUMMARIZATION_MODEL,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONF_VIDEO_ANALYZER_MODE,
    EDGE_CHAT_MODEL_URL,
    EMBEDDING_MODEL_CTX,
    EMBEDDING_MODEL_DIMS,
    EMBEDDING_MODEL_URL,
    RECOMMENDED_EDGE_CHAT_MODEL,
    RECOMMENDED_EMBEDDING_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    RECOMMENDED_VLM,
    SUMMARIZATION_MODEL_CTX,
    SUMMARIZATION_MODEL_PREDICT,
    SUMMARIZATION_MODEL_REASONING_DELIMITER,
    SUMMARIZATION_MODEL_URL,
    VIDEO_ANALYZER_DELETE_SNAPSHOTS,
    VIDEO_ANALYZER_MOBILE_APP,
    VIDEO_ANALYZER_PROMPT,
    VIDEO_ANALYZER_SCAN_INTERVAL,
    VIDEO_ANALYZER_SIMILARITY_THRESHOLD,
    VIDEO_ANALYZER_SNAPSHOT_ROOT,
    VIDEO_ANALYZER_SYSTEM_MESSAGE,
    VIDEO_ANALYZER_TIME_OFFSET,
    VLM_URL,
    CONF_DB_HOST,
    CONF_DB_PORT,
    CONF_DB_NAME,
    CONF_DB_USER,
    CONF_DB_PASSWORD,
    CONF_DB_SSLMODE,
    DEFAULT_DB_HOST,
    DEFAULT_DB_PORT,
    DEFAULT_DB_NAME,
    DEFAULT_DB_USER,
    DEFAULT_DB_PASSWORD,
    DEFAULT_DB_SSLMODE,
)
from .tools import analyze_image

LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = (Platform.CONVERSATION,)

type HGAConfigEntry = ConfigEntry[HGAData]

@dataclass
class HGAData:
    """Data for Home Generative Assistant."""

    chat_model: ChatOpenAI
    edge_chat_model: ChatOllama
    vision_model: ChatOllama
    summarization_model: ChatOllama
    pool: AsyncConnectionPool
    store: AsyncPostgresStore
    video_analyzer: VideoAnalyzer

async def _generate_embeddings(
        texts: list[str],
        model: OllamaEmbeddings
    ) -> list[list[float]]:
    """Generate embeddings from a list of text."""
    return await model.aembed_documents(texts)

class VideoAnalyzer:
    """Analyze video from recording cameras."""

    def __init__(self, hass: HomeAssistant, entry: HGAConfigEntry) -> None:
        """Init the video analyzer."""
        # Track snapshots and pending writes per camera.
        self.camera_snapshots: dict[str, list[Path]] = {}
        self.camera_write_locks: dict[str, asyncio.Lock] = {}

        self.hass = hass
        self.entry = entry

    @callback
    def _get_recording_cameras(self) -> list[str]:
        """Return a list of cameras currently recording."""
        return [
            state.entity_id for state in self.hass.states.async_all("camera")
            if state.state == "recording"
        ]

    async def _take_snapshot(self, now: datetime) -> None:
        """Take snapshots from all recording cameras."""
        snapshot_root_path = Path(VIDEO_ANALYZER_SNAPSHOT_ROOT)
        snapshot_root_path.mkdir(parents=True, exist_ok=True)

        for camera_id in self._get_recording_cameras():
            timestamp = dt_util.as_local(now).strftime("%Y%m%d_%H%M%S")
            cam_dir = snapshot_root_path / camera_id.replace(".", "_")
            cam_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = cam_dir / f"snapshot_{timestamp}.jpg"

            # Create a lock to track pending snapshot write.
            lock = self.camera_write_locks.setdefault(camera_id, asyncio.Lock())

            async with lock:
                await self.hass.services.async_call(
                    CAMERA_DOMAIN,
                    "snapshot",
                    {
                        "entity_id": camera_id,
                        "filename": str(snapshot_path),
                    },
                    blocking=True,
                )
                self.camera_snapshots.setdefault(camera_id, []).append(snapshot_path)
                LOGGER.debug("[%s] Snapshot saved to %s", camera_id, snapshot_path)

    async def _send_notification(
            self,
            msg: str,
            camera_name: str,
            camera_id: str,
            notify_img_path: Path
        ) -> None:
        """Send notification to the mobile app."""
        await self.hass.services.async_call(
            "notify",
            VIDEO_ANALYZER_MOBILE_APP,
            {
                "message": msg,
                "title": f"Camera Alert from {camera_name}!",
                "data": {
                    "entity_id:": camera_id,
                    "image": str(notify_img_path)
                }
            },
            blocking=True
        )

    async def _generate_summary(
            self,
            frame_descriptions: list[str],
            camera_id: str
        ) -> str:
        """Generate video scene summary analysis from its frame descriptions."""
        if not frame_descriptions:
            return ValueError("There must be at least one frame description.")

        if len(frame_descriptions) == 1:
            return frame_descriptions[0]

        options = self.entry.options
        prompt_start = VIDEO_ANALYZER_PROMPT
        tag_template = "\n<frame description>\n{i}\n</frame description>"
        prompt_parts = [tag_template.format(i=i) for i in frame_descriptions]
        prompt_parts.insert(0, prompt_start)
        prompt = " ".join(prompt_parts)
        LOGGER.debug("Prompt: %s", prompt)
        system_message = VIDEO_ANALYZER_SYSTEM_MESSAGE
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        model = self.entry.summarization_model
        model_with_config = model.with_config(
            config={
                "model": options.get(
                    CONF_SUMMARIZATION_MODEL,
                    RECOMMENDED_SUMMARIZATION_MODEL,
                ),
                "temperature": options.get(
                    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
                ),
                "top_p": options.get(
                    CONF_SUMMARIZATION_MODEL_TOP_P,
                    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
                ),
                "num_predict": SUMMARIZATION_MODEL_PREDICT,
                "num_ctx": SUMMARIZATION_MODEL_CTX,
            }
        )
        model_response = await model_with_config.ainvoke(messages)
        summary: str = model_response.content
        LOGGER.debug("Summary for %s: %s", camera_id, summary)
        # If model used reasoning, just return the final result.
        first, sep, last = summary.partition(
            SUMMARIZATION_MODEL_REASONING_DELIMITER.get("end", "")
        )
        if sep:
            return last.strip("\n")
        return first.strip("\n")

    async def _is_anomaly(
            self,
            camera_name: str,
            msg: str,
            img_path_parts: list[str]
        ) -> bool:
        """Perform anomaly detection on video analysis."""
        # Sematic search of the store with the video analysis as query.
        search_results = await self.entry.store.asearch(
            ("video_analysis", camera_name),
            query=msg,
            limit=10
        )
        LOGGER.debug("Search results: %s", search_results)

        # Calculate a "no newer than" time threshold from first snapshot time
        # by delaying it by the time offset.
        # Snapshot names are in the form "snapshot_20250426_002804.jpg".
        first_str = img_path_parts[-1].replace("snapshot_", "").replace(".jpg", "")
        first_dt = dt_util.as_local(datetime.strptime(first_str, "%Y%m%d_%H%M%S"))  # noqa: DTZ007
        no_newer_dt = first_dt - timedelta(seconds=VIDEO_ANALYZER_TIME_OFFSET)

        # Simple anomaly detection.
        # If all search results are older then the time threshold or if any are
        # newer or equal to it, and have a lower score then the similarity
        # threshold, declare the current video analysis as an anomaly.
        return (
            all(r.updated_at < no_newer_dt for r in search_results) or
            any(r.updated_at >= no_newer_dt and
                r.score < VIDEO_ANALYZER_SIMILARITY_THRESHOLD
                for r in search_results)
        )

    async def _process_snapshots(self, camera_id: str) -> None:
        """Process snapshots after a camera stops recording."""
        lock = self.camera_write_locks.get(camera_id)
        if lock:
            LOGGER.debug("[%s] Waiting for snapshot writes to finish...", camera_id)
            async with lock:
                LOGGER.debug("[%s] Done waiting for writes.", camera_id)

        snapshots = self.camera_snapshots.get(camera_id, [])
        if not snapshots:
            return

        camera_name = camera_id.split(".")[-1]

        options = self.entry.options

        LOGGER.debug("[%s] Processing %s snapshots...", camera_id, len(snapshots))
        frame_descriptions: list[str] = []
        for path in snapshots:
            LOGGER.debug(" - %s", path)

            async with aiofiles.open(path, "rb") as file:
                image = await file.read()

                detection_keywords = None
                frame_description = await analyze_image(
                    self.entry.vision_model, options, image, detection_keywords
                )
                LOGGER.debug("Analysis for %s: %s", path, frame_description)
                frame_descriptions.append(frame_description)

        msg = await self._generate_summary(frame_descriptions, camera_id)

        # Grab first snapshot image path parts.
        img_path_parts = snapshots[0].parts

        # First snapshot path to use as a static image in the mobile app notification.
        notify_img_path = Path("/media/local") / Path(*img_path_parts[-3:])

        if (mode := options.get(CONF_VIDEO_ANALYZER_MODE)) == "notify_on_anomaly":
            is_anomaly = await self._is_anomaly(camera_name, msg, img_path_parts)
            LOGGER.debug("Is anomaly: %s", is_anomaly)

            if is_anomaly:
                await self._send_notification(
                    msg, camera_name, camera_id, notify_img_path
                )
        elif mode == "always_notify":
            await self._send_notification(msg, camera_name, camera_id, notify_img_path)

        # Store current msg and associated snapshots.
        await self.entry.store.aput(
            namespace=("video_analysis", camera_name),
            key=img_path_parts[-1], # key is date and time of first snapshot
            value={"content": msg, "snapshots": [str(s) for s in snapshots]},
        )

        # Clean-up.
        self.camera_snapshots[camera_id] = []
        if VIDEO_ANALYZER_DELETE_SNAPSHOTS:
            for path in snapshots:
                try:
                    path.unlink()
                except OSError:
                    LOGGER.warning("Failed to delete snapshot: %s", path)

    @callback
    def _handle_camera_state_change(self, event: Event) -> None:
        """Handle camera state changes to trigger processing."""
        entity_id = event.data.get("entity_id")
        if not entity_id or not entity_id.startswith("camera."):
            return

        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")
        if old_state is None or new_state is None:
            return

        if old_state.state == "recording" and new_state.state != "recording":
            # Debounce: wait 1 second before processing.
            async def _delayed_process() -> None:
                await asyncio.sleep(1)
                await self._process_snapshots(entity_id)

            self.hass.async_create_task(_delayed_process())

    def start(self) -> None:
        """Start the video analyzer."""
        # Start video analyzer snapshot job.
        self.cancel_track = async_track_time_interval(
            self.hass,
            self._take_snapshot,
            timedelta(seconds=VIDEO_ANALYZER_SCAN_INTERVAL)
        )
        # Watch for recording cameras and analyze video.
        self.cancel_listen = self.hass.bus.async_listen(
            EVENT_STATE_CHANGED,
            self._handle_camera_state_change
        )
        LOGGER.info("Video analyzer started.")

    def stop(self) -> None:
        """Stop the video analyzer."""
        if self.is_running():
            self.cancel_track()
            self.cancel_listen()
            LOGGER.info("Video analyzer stopped.")

    def is_running(self) -> bool:
        """Check if video analyzer is running."""
        return hasattr(self, "cancel_track") and hasattr(self, "cancel_listen")

async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Set up Home Generative Agent from a config entry."""
    # Initialize models and verify they were setup correctly.
    chat_model = ChatOpenAI(
        api_key=entry.data.get(CONF_API_KEY),
        timeout=10,
        http_async_client=get_async_client(hass),
    ).configurable_fields(
        model_name=ConfigurableField(id="model_name"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        max_tokens=ConfigurableField(id="max_tokens"),
    )
    try:
        await hass.async_add_executor_job(chat_model.get_name)
    except HomeAssistantError:
        LOGGER.exception("Error setting up chat model")
        return False
    entry.chat_model = chat_model

    edge_chat_model = ChatOllama(
        model=RECOMMENDED_EDGE_CHAT_MODEL,
        base_url=EDGE_CHAT_MODEL_URL,
        http_async_client=get_async_client(hass)
    ).configurable_fields(
        model=ConfigurableField(id="model"),
        format=ConfigurableField(id="format"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        num_predict=ConfigurableField(id="num_predict"),
        num_ctx=ConfigurableField(id="num_ctx"),
    )
    try:
        await hass.async_add_executor_job(edge_chat_model.get_name)
    except HomeAssistantError:
        LOGGER.exception("Error setting up edge chat model")
        return False
    entry.edge_chat_model = edge_chat_model

    vision_model = ChatOllama(
        model=RECOMMENDED_VLM,
        base_url=VLM_URL,
        http_async_client=get_async_client(hass)
    ).configurable_fields(
        model=ConfigurableField(id="model"),
        format=ConfigurableField(id="format"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        num_predict=ConfigurableField(id="num_predict"),
        num_ctx=ConfigurableField(id="num_ctx"),
    )
    try:
        await hass.async_add_executor_job(vision_model.get_name)
    except HomeAssistantError:
        LOGGER.exception("Error setting up VLM")
        return False
    entry.vision_model = vision_model

    summarization_model = ChatOllama(
        model=RECOMMENDED_SUMMARIZATION_MODEL,
        base_url=SUMMARIZATION_MODEL_URL,
        http_async_client=get_async_client(hass)
    ).configurable_fields(
        model=ConfigurableField(id="model"),
        format=ConfigurableField(id="format"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        num_predict=ConfigurableField(id="num_predict"),
        num_ctx=ConfigurableField(id="num_ctx"),
    )
    try:
        await hass.async_add_executor_job(vision_model.get_name)
    except HomeAssistantError:
        LOGGER.exception("Error setting up summarization model")
        return False
    entry.summarization_model = summarization_model

    embedding_model = OllamaEmbeddings(
        model=RECOMMENDED_EMBEDDING_MODEL,
        base_url=EMBEDDING_MODEL_URL,
        num_ctx=EMBEDDING_MODEL_CTX
    )
    entry.embedding_model = embedding_model

    # Open postgresql database for short-term and long-term memory.
    host = entry.options.get(CONF_DB_HOST, DEFAULT_DB_HOST)
    port = entry.options.get(CONF_DB_PORT, DEFAULT_DB_PORT)
    name = entry.options.get(CONF_DB_NAME, DEFAULT_DB_NAME)
    user = entry.options.get(CONF_DB_USER, DEFAULT_DB_USER)
    password = entry.options.get(CONF_DB_PASSWORD, DEFAULT_DB_PASSWORD)
    sslmode = (
        "disable"
        if not entry.options.get(CONF_DB_SSLMODE, DEFAULT_DB_SSLMODE)
        else "require"
    )

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{name}?sslmode={sslmode}"
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row
    }
    pool = AsyncConnectionPool(
        conninfo=dsn,
        min_size=5,
        max_size=20,
        kwargs=connection_kwargs,
        open=False
    )
    try:
        await pool.open()
    except PoolTimeout:
        LOGGER.exception("Error opening postgresql db")
        return False
    entry.pool = pool

    # Initialize store for session-based (long-term) memory with semantic search.
    store = AsyncPostgresStore(
        pool,
        index={
            "embed": partial(
                _generate_embeddings,
                model=embedding_model
            ),
            "dims": EMBEDDING_MODEL_DIMS,
            "fields": ["content"]
        }
    )
    entry.store = store

    # Initialize video analyzer and start if option is set.
    video_analyzer = VideoAnalyzer(hass, entry)
    if entry.options.get(CONF_VIDEO_ANALYZER_MODE) != "disable":
        video_analyzer.start()
    entry.video_analyzer = video_analyzer

    # Setup conversation platform.
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Unload Home Generative Agent."""
    pool = entry.pool
    await pool.close()

    video_analyzer = entry.video_analyzer
    video_analyzer.stop()

    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    return True
