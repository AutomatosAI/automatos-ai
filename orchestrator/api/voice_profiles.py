"""
Voice Profile API Endpoints (PRD-74 Phase 2)
=============================================

REST APIs for voice profile CRUD, preview synthesis, and voice cloning:
- List workspace voice profiles
- Create / update / delete profiles
- Generate preview audio from a profile
- Upload reference audio for voice cloning
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import boto3
from botocore.config import Config as BotoConfig
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["Voice Profiles"])

ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm"}
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm"}
MIN_REFERENCE_SECONDS = 5
MAX_REFERENCE_SECONDS = 60
PREVIEW_TEXT = "Hello, this is a preview of how I will sound when speaking to you."


# ===================================================================
# Pydantic Models
# ===================================================================

class VoiceProfileOut(BaseModel):
    id: str
    workspace_id: str
    name: str
    provider: str
    voice_id: str
    reference_audio: Optional[str] = None
    settings: Dict = {}
    is_default: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class CreateVoiceProfileBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    provider: str = Field("kokoro", max_length=100)
    voice_id: str = Field(..., min_length=1, max_length=255)
    settings: Dict = Field(default_factory=dict)
    is_default: bool = False


class UpdateVoiceProfileBody(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    provider: Optional[str] = Field(None, max_length=100)
    voice_id: Optional[str] = Field(None, min_length=1, max_length=255)
    settings: Optional[Dict] = None
    is_default: Optional[bool] = None


# ===================================================================
# Helpers
# ===================================================================

def _profile_to_out(p) -> VoiceProfileOut:
    return VoiceProfileOut(
        id=str(p.id),
        workspace_id=str(p.workspace_id),
        name=p.name,
        provider=p.provider,
        voice_id=p.voice_id,
        reference_audio=p.reference_audio,
        settings=p.settings or {},
        is_default=p.is_default if p.is_default is not None else False,
        created_at=p.created_at.isoformat() if p.created_at else None,
        updated_at=p.updated_at.isoformat() if p.updated_at else None,
    )


def _get_s3_client():
    """Build an S3 client using centralized config."""
    boto_cfg = BotoConfig(
        region_name=config.AWS_REGION,
        signature_version="s3v4",
    )
    if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
        return boto3.client(
            "s3",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            config=boto_cfg,
        )
    return boto3.client("s3", config=boto_cfg)


def _validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file type and extension."""
    # Check extension
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Content type check (best-effort — browsers may send generic types)
    ct = (file.content_type or "").lower()
    if ct and ct != "application/octet-stream" and ct not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{ct}'. Allowed: wav, mp3, webm",
        )


async def _validate_audio_duration(audio_bytes: bytes) -> None:
    """Estimate audio duration and reject out-of-range files.

    Uses a rough size-based heuristic. For wav files we can parse the
    header; for compressed formats we fall back to bitrate estimation.
    """
    size = len(audio_bytes)

    # WAV header parsing: bytes 28-31 = byte rate
    if size > 44 and audio_bytes[:4] == b"RIFF":
        byte_rate = int.from_bytes(audio_bytes[28:32], "little")
        if byte_rate > 0:
            duration_s = (size - 44) / byte_rate
            if duration_s < MIN_REFERENCE_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio too short ({duration_s:.1f}s). Minimum {MIN_REFERENCE_SECONDS}s.",
                )
            if duration_s > MAX_REFERENCE_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio too long ({duration_s:.1f}s). Maximum {MAX_REFERENCE_SECONDS}s.",
                )
            return

    # For mp3/webm: estimate at ~128kbps
    estimated_duration_s = size / (128_000 / 8)
    if estimated_duration_s < MIN_REFERENCE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio appears too short (~{estimated_duration_s:.0f}s). Minimum {MIN_REFERENCE_SECONDS}s.",
        )
    if estimated_duration_s > MAX_REFERENCE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio appears too long (~{estimated_duration_s:.0f}s). Maximum {MAX_REFERENCE_SECONDS}s.",
        )


# ===================================================================
# Endpoints
# ===================================================================

@router.get("/profiles", response_model=None)
async def list_voice_profiles(
    provider: Optional[str] = Query(None, description="Filter by TTS provider"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all voice profiles for the workspace."""
    try:
        from core.models.voice_profiles import VoiceProfile

        query = db.query(VoiceProfile).filter(
            VoiceProfile.workspace_id == ctx.workspace_id,
        )
        if provider:
            query = query.filter(VoiceProfile.provider == provider)

        query = query.order_by(VoiceProfile.is_default.desc(), VoiceProfile.name)
        profiles = query.all()

        return {
            "items": [_profile_to_out(p).model_dump() for p in profiles],
            "total": len(profiles),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing voice profiles: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/profiles", status_code=201, response_model=VoiceProfileOut)
async def create_voice_profile(
    body: CreateVoiceProfileBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new voice profile for the workspace."""
    try:
        from core.models.voice_profiles import VoiceProfile

        # If marking as default, unset existing default for this workspace
        if body.is_default:
            db.query(VoiceProfile).filter(
                VoiceProfile.workspace_id == ctx.workspace_id,
                VoiceProfile.is_default == True,
            ).update({"is_default": False}, synchronize_session="fetch")

        profile = VoiceProfile(
            workspace_id=ctx.workspace_id,
            name=body.name,
            provider=body.provider,
            voice_id=body.voice_id,
            settings=body.settings,
            is_default=body.is_default,
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)

        return _profile_to_out(profile)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating voice profile: %s", e, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/profiles/{profile_id}", response_model=VoiceProfileOut)
async def get_voice_profile(
    profile_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get a single voice profile by ID."""
    try:
        from core.models.voice_profiles import VoiceProfile

        profile = db.query(VoiceProfile).filter(
            VoiceProfile.id == profile_id,
            VoiceProfile.workspace_id == ctx.workspace_id,
        ).first()

        if not profile:
            raise HTTPException(status_code=404, detail="Voice profile not found")

        return _profile_to_out(profile)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching voice profile %s: %s", profile_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/profiles/{profile_id}", response_model=VoiceProfileOut)
async def update_voice_profile(
    profile_id: UUID,
    body: UpdateVoiceProfileBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update an existing voice profile's settings."""
    try:
        from core.models.voice_profiles import VoiceProfile

        profile = db.query(VoiceProfile).filter(
            VoiceProfile.id == profile_id,
            VoiceProfile.workspace_id == ctx.workspace_id,
        ).first()

        if not profile:
            raise HTTPException(status_code=404, detail="Voice profile not found")

        # If marking as default, unset existing default
        if body.is_default is True:
            db.query(VoiceProfile).filter(
                VoiceProfile.workspace_id == ctx.workspace_id,
                VoiceProfile.is_default == True,
                VoiceProfile.id != profile_id,
            ).update({"is_default": False}, synchronize_session="fetch")

        if body.name is not None:
            profile.name = body.name
        if body.provider is not None:
            profile.provider = body.provider
        if body.voice_id is not None:
            profile.voice_id = body.voice_id
        if body.settings is not None:
            profile.settings = body.settings
        if body.is_default is not None:
            profile.is_default = body.is_default

        db.commit()
        db.refresh(profile)

        return _profile_to_out(profile)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating voice profile %s: %s", profile_id, e, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/profiles/{profile_id}")
async def delete_voice_profile(
    profile_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a voice profile. Clears agent assignments referencing it."""
    try:
        from core.models.voice_profiles import VoiceProfile

        profile = db.query(VoiceProfile).filter(
            VoiceProfile.id == profile_id,
            VoiceProfile.workspace_id == ctx.workspace_id,
        ).first()

        if not profile:
            raise HTTPException(status_code=404, detail="Voice profile not found")

        # Clear agent references
        from core.models.core import Agent
        db.query(Agent).filter(
            Agent.workspace_id == ctx.workspace_id,
            Agent.voice_profile_id == profile_id,
        ).update({"voice_profile_id": None}, synchronize_session="fetch")

        profile_name = profile.name
        db.delete(profile)
        db.commit()

        return {
            "success": True,
            "message": f"Voice profile '{profile_name}' deleted",
            "profile_id": str(profile_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting voice profile %s: %s", profile_id, e, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/profiles/{profile_id}/preview")
async def preview_voice_profile(
    profile_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Generate a short audio preview using the profile's voice settings."""
    try:
        from core.models.voice_profiles import VoiceProfile

        profile = db.query(VoiceProfile).filter(
            VoiceProfile.id == profile_id,
            VoiceProfile.workspace_id == ctx.workspace_id,
        ).first()

        if not profile:
            raise HTTPException(status_code=404, detail="Voice profile not found")

        from modules.voice.client import VoiceServiceClient
        voice_client = VoiceServiceClient()

        speed = (profile.settings or {}).get("speed", 1.0)

        result = await voice_client.synthesize(
            text=PREVIEW_TEXT,
            voice=profile.voice_id,
            speed=speed,
            response_format="mp3",
            model=profile.provider,
            reference_audio=profile.reference_audio,
        )

        import base64
        return {
            "audio_base64": base64.b64encode(result.audio).decode("ascii"),
            "format": result.format,
            "duration_ms": result.duration_ms,
            "profile_id": str(profile_id),
            "voice_id": profile.voice_id,
            "provider": profile.provider,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error previewing voice profile %s: %s", profile_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Voice preview failed")


@router.post("/profiles/clone", status_code=201, response_model=VoiceProfileOut)
async def clone_voice_profile(
    name: str = Query(..., min_length=1, max_length=255, description="Profile name"),
    provider: str = Query("chatterbox", max_length=100, description="TTS provider"),
    file: UploadFile = File(..., description="Reference audio file (wav/mp3/webm, 5-60s)"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Upload reference audio and create a cloned voice profile.

    Accepts wav, mp3, or webm. Audio must be between 5 and 60 seconds.
    The file is stored in S3 and a voice profile record is created.
    """
    _validate_audio_file(file)

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    await _validate_audio_duration(audio_bytes)

    try:
        from core.models.voice_profiles import VoiceProfile

        profile_id = uuid4()
        s3_key = f"workspaces/{ctx.workspace_id}/voices/{profile_id}/reference.wav"

        # Upload to S3
        s3 = _get_s3_client()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: s3.put_object(
                Bucket=config.S3_DOCUMENTS_BUCKET,
                Key=s3_key,
                Body=audio_bytes,
                ContentType=file.content_type or "audio/wav",
            ),
        )

        profile = VoiceProfile(
            id=profile_id,
            workspace_id=ctx.workspace_id,
            name=name,
            provider=provider,
            voice_id=f"clone_{profile_id.hex[:12]}",
            reference_audio=s3_key,
            settings={"source_filename": file.filename or "unknown"},
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)

        logger.info(
            "voice_clone_created",
            extra={
                "profile_id": str(profile_id),
                "workspace_id": str(ctx.workspace_id),
                "s3_key": s3_key,
                "audio_bytes": len(audio_bytes),
            },
        )

        return _profile_to_out(profile)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error cloning voice profile: %s", e, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Voice cloning failed")
