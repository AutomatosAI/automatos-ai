"""PRD-273 test helpers: a workspace with something in every category, and secrets planted where
no export may carry them.

``seed_workspace_contents`` writes, inside the caller's session (rolled back or swept by the
caller): the workspace's settings (a brand kit, an Orchestrator block with a stale soul and a
model, BYOK keys and integrations), Auto's agent row with its real soul, a second agent with a
global and a workspace skill, an enabled global skill, a team, a chat with a message, a
short-term memory, three documents (two with the same bytes, one outside the workspace's
storage), a generated Deliverable, a board ticket, a Playbook with a webhook trigger (its id and
signing secret), an agent configuration holding a pasted key, and rows of NOT_MOVED tables: a user
API key, an SDK key, a paired machine. Every secret is a fresh
random value, listed in ``Seeded.secrets``; ``files`` is what each object holds.
"""
from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from sqlalchemy import text

from core.models import Agent, Skill
from core.models.core import BoardTask, Chat, Document, Message, Team, UserApiKey, WorkflowTemplate, agent_skills
from modules.documents.thumbnails.store import documents_bucket
from modules.memory.models import MemoryShortTerm

AUTO_SOUL = "You are Auto, the owner's chief of staff. Plain words, short answers."
STALE_SOUL = "an old copy of the soul that the settings page no longer shows"
DOCUMENT_BYTES = b"%PDF-1.4 the quarterly plan"
OTHER_BYTES = b"a second file, a generated invoice"


@dataclass
class Seeded:
    workspace_id: str
    user_id: int
    auto_id: int
    writer_id: int
    playbook_id: int
    global_skill: str
    secrets: List[str] = field(default_factory=list)
    objects: Dict[str, bytes] = field(default_factory=dict)


def _secret() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex


def seed_user(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd273-{tag}", "e": f"prd273-{tag}@example.test"},
    ).scalar()


def _settings(seeded: Seeded) -> dict:
    byok, integration = _secret(), _secret()
    seeded.secrets.extend([byok, integration])
    return {
        "brand_kit": {"primary_color": "#1F6FEB", "voice": {"sign_off": "The Harbour Team"}},
        "orchestrator": {"custom_soul": STALE_SOUL, "communication_style": "direct",
                         "llm": {"model_id": "openai/gpt-5"}, "model": "openai/gpt-5"},
        "voice_live": {"voice": "warm"}, "push": {"digest_time": "08:00"},
        "byok_overrides": {"openai": byok}, "integrations": {"slack": {"bot_token": integration}},
    }


def _agents(db, ws: uuid.UUID, seeded: Seeded) -> None:
    pasted = _secret()
    seeded.secrets.append(pasted)
    auto = Agent(name="Auto", agent_type="system", workspace_id=ws, slug=f"auto-{ws}", is_system_agent=True,
                 custom_persona_prompt=AUTO_SOUL, use_custom_persona=True,
                 configuration={"personality_mode": "custom", "proactive_level": "notify"})
    writer = Agent(name="Writer", agent_type="custom", workspace_id=ws, custom_persona_prompt="You write.",
                   use_custom_persona=True, model_config={"model_id": "anthropic/claude-haiku-5-5", "provider": "anthropic"},
                   configuration={"tone": "plain", "tools": {"search": {"api_key": pasted}}})
    shared = Skill(name=seeded.global_skill, skill_type="prompt", workspace_id=None)
    own = Skill(name=f"own-{uuid.uuid4().hex[:8]}", skill_type="prompt", workspace_id=ws)
    db.add_all([auto, writer, shared, own])
    db.flush()
    db.execute(agent_skills.insert(), [{"agent_id": writer.id, "skill_id": shared.id},
                                       {"agent_id": writer.id, "skill_id": own.id}])
    db.execute(text("INSERT INTO workspace_enabled_skills (workspace_id, skill_id) VALUES (:ws, :skill)"),
               {"ws": str(ws), "skill": shared.id})
    seeded.auto_id, seeded.writer_id = auto.id, writer.id


def _playbook(db, ws: uuid.UUID, seeded: Seeded) -> None:
    webhook, signing = _secret(), _secret()
    seeded.secrets.extend([webhook, signing])
    playbook = WorkflowTemplate(
        template_id=f"prd273-{uuid.uuid4().hex[:10]}", name="Weekly digest", description="Every Monday.",
        workspace_id=ws, owner_type="workspace", template_definition={}, created_by="test",
        steps=[{"step_id": "s1", "order": 1, "type": "agent", "agent_id": seeded.writer_id,
                "prompt_template": "Write the digest."}],
        schedule_config={"type": "webhook", "webhook_id": webhook, "webhook_secret": signing},
    )
    db.add(playbook)
    db.flush()
    seeded.playbook_id = playbook.id


def _conversation(db, ws: uuid.UUID, seeded: Seeded) -> None:
    team = Team(workspace_id=ws, name="Growth", normalized_name="growth")
    chat = Chat(user_id=seeded.user_id, workspace_id=ws, title="Plans")
    db.add_all([team, chat, MemoryShortTerm(workspace_id=ws, content="The owner prefers mornings.")])
    db.flush()
    db.add(Message(chat_id=chat.id, workspace_id=ws, role="user", parts=[{"type": "text", "text": "Hello"}]))
    db.add(BoardTask(workspace_id=ws, title="Draft the newsletter"))
    db.flush()


def _documents(db, ws: uuid.UUID, seeded: Seeded, storage_dir: Path) -> None:
    bucket = documents_bucket()
    first, second = f"workspaces/{ws}/documents/1_plan.pdf", f"workspaces/{ws}/documents/2_plan.pdf"
    seeded.objects = {first: DOCUMENT_BYTES, second: DOCUMENT_BYTES}
    db.add_all([
        Document(filename="plan.pdf", original_filename="plan.pdf", file_path=f"s3://{bucket}/{first}", workspace_id=ws),
        Document(filename="copy.pdf", original_filename="copy.pdf", file_path=f"s3://{bucket}/{second}", workspace_id=ws),
        Document(filename="theirs.pdf", file_path=f"s3://{bucket}/workspaces/{uuid.uuid4()}/documents/9_x.pdf",
                 workspace_id=ws),
    ])
    generated = storage_dir / str(ws) / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "invoice.pdf").write_bytes(OTHER_BYTES)
    db.execute(text(
        "INSERT INTO deliverables (workspace_id, source_type, artifact_type, title, storage_type, file_path, file_name) "
        "VALUES (:ws, 'tool', 'document', 'Invoice', 'generated', 'generated/invoice.pdf', 'invoice.pdf')"),
        {"ws": str(ws)})
    db.flush()


def _not_moved(db, ws: uuid.UUID, seeded: Seeded) -> None:
    api_key, sdk_hash, host_hash = _secret(), _secret()[:64], _secret()[:64]
    seeded.secrets.extend([api_key, sdk_hash, host_hash])
    db.add(UserApiKey(workspace_id=ws, provider="openai", encrypted_key=api_key))
    db.execute(text(
        "INSERT INTO sdk_api_keys (id, workspace_id, name, key_prefix, key_hash, key_type) "
        "VALUES (gen_random_uuid(), :ws, 'site key', 'ak_', :hash, 'server')"), {"ws": str(ws), "hash": sdk_hash})
    db.execute(text(
        "INSERT INTO cli_hosts (id, workspace_id, name, token_hash) VALUES (gen_random_uuid(), :ws, 'laptop', :hash)"),
        {"ws": str(ws), "hash": host_hash})
    db.flush()


def seed_workspace_contents(db, workspace_id: str, storage_dir: Path) -> Seeded:
    """Fill ``workspace_id`` (an existing workspaces row) as the module docstring says."""
    ws = uuid.UUID(str(workspace_id))
    seeded = Seeded(workspace_id=str(ws), user_id=seed_user(db), auto_id=0, writer_id=0, playbook_id=0,
                    global_skill=f"global-{uuid.uuid4().hex[:8]}")
    db.execute(text("UPDATE workspaces SET settings = CAST(:settings AS jsonb) WHERE id = :ws"),
               {"settings": _json(_settings(seeded)), "ws": str(ws)})
    _agents(db, ws, seeded)
    _playbook(db, ws, seeded)
    _conversation(db, ws, seeded)
    _documents(db, ws, seeded, storage_dir)
    _not_moved(db, ws, seeded)
    return seeded


def _json(value) -> str:
    import json

    return json.dumps(value)


class FakeObjects:
    """The slice of an S3 client the export reads and writes, over a dict of objects."""

    def __init__(self, objects: Dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.deleted: List[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3's names
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def upload_fileobj(self, fileobj, Bucket, Key):  # noqa: N803
        self.objects[Key] = fileobj.read()

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.deleted.append(Key)
        self.objects.pop(Key, None)
