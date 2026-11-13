"""Partner tool packs: the format (PRD-275).

A pack is a JSON document an operator installs into a workspace. It declares read-only tools
on a partner's own HTTP API. Each tool is a small set of GET operations; each operation is a
path template, the parameters the model may send, and the part of the answer to keep. A pack
carries no host and no key: those are the workspace's credential (``store``), so one pack
serves a development and a production install, and the model never sees either.

There is no method field: every operation is a GET. An unknown field is refused, so a pack
that asks for anything else does not install.

This module is pure. It checks a pack and builds the tool schemas the model is offered.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_ID = "automatos-partner-pack/v1"
# The argument that picks one of a tool's operations; no parameter may take its name.
OPERATION_KEY = "operation"
# A pack's tools are offered as "<pack>_<tool>": the pack is the part before the first underscore.
# Together they stay within the 64 characters a model provider allows a tool name.
PACK_NAME = r"^[a-z][a-z0-9]{1,19}$"
TOOL_NAME = r"^[a-z][a-z0-9_]{1,40}$"
OPERATION_NAME = r"^[a-z][a-z0-9_]{0,30}$"
VERSION = r"^\d+\.\d+\.\d+$"
FULL_TOOL_NAME = re.compile(r"([a-z][a-z0-9]{1,19})_([a-z][a-z0-9_]{1,40})")
PARAMETER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,40}")
FIELD_PATH = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*")
# Literal segments and {parameter} placeholders: no host, no query, no dot segments.
PATH = re.compile(r"(/([A-Za-z0-9_~-][A-Za-z0-9._~-]*|\{[A-Za-z][A-Za-z0-9_]{0,40}\}))+")
PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_]{0,40})\}")
# Names the platform's own tools start with: a pack may not take one.
RESERVED_PACK_NAMES = frozenset({"platform", "workspace", "composio", "widget"})

DEFAULT_MAX_ITEMS = 25
MAX_ITEMS_CEILING = 200
# The chat keeps the first 2,000 tokens of a tool's answer (consumers.chatbot.tool_cut). The caps
# here are sized to stay under that, so an answer is cut to whole records before the chat can cut
# one in half.
DEFAULT_MAX_CHARS = 4_000
MAX_CHARS_CEILING = 6_000
MAX_PACK_CHARS = 200_000
ERRORS_SHOWN = 8

Scalar = Union[bool, int, float, str]


class PackError(ValueError):
    """A pack that cannot be installed; the message says what to change."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Parameter(_Strict):
    """One value the model may send: a scalar, described for the model."""

    type: Literal["string", "integer", "number", "boolean"]
    description: str = Field(default="", max_length=500)
    enum: Optional[Tuple[Scalar, ...]] = Field(default=None, min_length=1, max_length=50)
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    minLength: Optional[int] = Field(default=None, ge=0, le=1000)
    maxLength: Optional[int] = Field(default=None, ge=1, le=1000)

    def as_json_schema(self) -> Dict[str, Any]:
        """The JSON Schema for this value: only what the pack set."""
        return {key: (list(value) if isinstance(value, tuple) else value)
                for key, value in self.model_dump().items() if value is not None and value != ""}


class Parameters(_Strict):
    """An operation's parameters, written as the JSON Schema object the model is shown."""

    type: Literal["object"] = "object"
    properties: Dict[str, Parameter] = Field(default_factory=dict, max_length=30)
    required: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def _names_are_declared(self) -> "Parameters":
        bad = [name for name in self.properties if not PARAMETER_NAME.fullmatch(name) or name == OPERATION_KEY]
        if bad:
            raise ValueError(f"parameter names {bad} are not allowed: use letters, digits and '_'; "
                             f"'{OPERATION_KEY}' is reserved")
        missing = [name for name in self.required if name not in self.properties]
        if missing:
            raise ValueError(f"required names {missing} are not in properties")
        return self

    def as_json_schema(self) -> Dict[str, Any]:
        """The schema a call's arguments are checked against, and the model is shown."""
        return {"type": "object",
                "properties": {name: spec.as_json_schema() for name, spec in self.properties.items()},
                "required": list(self.required)}


class ResponseShape(_Strict):
    """What of the partner's answer the model is given: where to look, the fields kept, the caps."""

    root: str = Field(default="", max_length=200)
    keep: Tuple[str, ...] = Field(min_length=1, max_length=60)
    max_items: int = Field(default=DEFAULT_MAX_ITEMS, ge=1, le=MAX_ITEMS_CEILING)
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, ge=200, le=MAX_CHARS_CEILING)

    @model_validator(mode="after")
    def _paths_are_plain(self) -> "ResponseShape":
        paths = [*self.keep, *([self.root] if self.root else [])]
        bad = [path for path in paths if not FIELD_PATH.fullmatch(path)]
        if bad:
            raise ValueError(f"field paths {bad} must be field names joined by dots")
        return self


class Operation(_Strict):
    """One GET on the partner's API."""

    name: str = Field(pattern=OPERATION_NAME)
    description: str = Field(min_length=1, max_length=600)
    path: str = Field(max_length=300)
    parameters: Parameters = Field(default_factory=Parameters)
    # Query values every call of this operation sends, whatever the model asks.
    query: Dict[str, Scalar] = Field(default_factory=dict, max_length=20)
    response: ResponseShape

    @model_validator(mode="after")
    def _path_and_parameters_agree(self) -> "Operation":
        if not PATH.fullmatch(self.path):
            raise ValueError("path must start with '/' and be literal segments and {parameter} "
                             "placeholders: no host, no query and no dot segments")
        for name in placeholders(self.path):
            spec = self.parameters.properties.get(name)
            if spec is None or name not in self.parameters.required or spec.type not in ("string", "integer"):
                raise ValueError(f"path parameter '{name}' must be a required string or integer parameter")
        clash = [key for key in self.query if key in self.parameters.properties or not PARAMETER_NAME.fullmatch(key)]
        if clash:
            raise ValueError(f"fixed query names {clash} must be plain names that are not also parameters")
        return self


class Tool(_Strict):
    """One tool the model is offered: a few operations on one kind of record."""

    name: str = Field(pattern=TOOL_NAME)
    description: str = Field(min_length=1, max_length=1000)
    # The permission the partner's key needs for this tool, said when the partner refuses the key.
    scope: str = Field(default="", max_length=100)
    operations: Tuple[Operation, ...] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def _operations_agree(self) -> "Tool":
        names = [operation.name for operation in self.operations]
        if len(set(names)) != len(names):
            raise ValueError(f"operation names must be unique, got {names}")
        seen: Dict[str, Parameter] = {}
        for operation in self.operations:
            for name, spec in operation.parameters.properties.items():
                if seen.setdefault(name, spec) != spec:
                    raise ValueError(f"parameter '{name}' is declared differently in two operations of '{self.name}'")
        return self


class PartnerPack(_Strict):
    """A partner's read-only tools, as an operator installs them."""

    schema_id: Literal["automatos-partner-pack/v1"] = Field(alias="schema")
    name: str = Field(pattern=PACK_NAME)
    title: str = Field(min_length=1, max_length=80)
    version: str = Field(pattern=VERSION)
    description: str = Field(default="", max_length=500)
    tools: Tuple[Tool, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _names_are_free(self) -> "PartnerPack":
        if self.name in RESERVED_PACK_NAMES:
            raise ValueError(f"'{self.name}' is a name the platform's own tools use; choose another")
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError(f"tool names must be unique, got {names}")
        return self

    def as_stored(self) -> Dict[str, Any]:
        """The pack as JSON, in the words it was written in."""
        return self.model_dump(mode="json", by_alias=True)


def placeholders(path: str) -> List[str]:
    """The parameter names a path template takes, in order."""
    return PLACEHOLDER.findall(path)


def parse_pack(data: Any) -> PartnerPack:
    """The pack ``data`` describes. Raises :class:`PackError` naming each fault."""
    if not isinstance(data, dict):
        raise PackError("A pack is a JSON object.")
    if len(json.dumps(data, default=str)) > MAX_PACK_CHARS:
        raise PackError(f"A pack is at most {MAX_PACK_CHARS} characters of JSON.")
    try:
        return PartnerPack.model_validate(data)
    except ValidationError as invalid:
        faults = [f"{'.'.join(str(part) for part in error['loc']) or 'pack'}: {error['msg']}"
                  for error in invalid.errors()[:ERRORS_SHOWN]]
        raise PackError("; ".join(faults)) from None


def full_tool_name(pack_name: str, tool: Tool) -> str:
    """The name the model calls the tool by."""
    return f"{pack_name}_{tool.name}"


def split_tool_name(name: Any) -> Optional[Tuple[str, str]]:
    """``(pack name, tool name)`` for a name shaped like a pack's tool; None for any other."""
    match = FULL_TOOL_NAME.fullmatch(name) if isinstance(name, str) else None
    return (match.group(1), match.group(2)) if match else None


def _takes(operation: Operation) -> str:
    required = set(operation.parameters.required)
    needs = [name for name in operation.parameters.properties if name in required]
    may = [name for name in operation.parameters.properties if name not in required]
    parts = [f"needs {', '.join(needs)}" if needs else "", f"may take {', '.join(may)}" if may else ""]
    said = "; ".join(part for part in parts if part)
    return f"{operation.name}: {operation.description}" + (f" ({said})" if said else "")


def _parameters_schema(tool: Tool) -> Dict[str, Any]:
    """One operation: its own parameters. Several: ``operation`` picks one, and every
    parameter is optional here because each operation is checked on its own when called."""
    if len(tool.operations) == 1:
        return tool.operations[0].parameters.as_json_schema()
    properties: Dict[str, Any] = {OPERATION_KEY: {
        "type": "string",
        "enum": [operation.name for operation in tool.operations],
        "description": " | ".join(_takes(operation) for operation in tool.operations),
    }}
    for operation in tool.operations:
        for name, spec in operation.parameters.properties.items():
            properties.setdefault(name, spec.as_json_schema())
    return {"type": "object", "properties": properties, "required": [OPERATION_KEY]}


def tool_schema(pack: PartnerPack, tool: Tool) -> Dict[str, Any]:
    """The tool in the function format the model is offered."""
    return {"type": "function", "function": {
        "name": full_tool_name(pack.name, tool),
        "description": f"{tool.description} Read-only: answers from {pack.title}.",
        "parameters": _parameters_schema(tool),
    }}


def pack_schemas(pack: PartnerPack) -> List[Dict[str, Any]]:
    """Every tool of the pack, as the model is offered them."""
    return [tool_schema(pack, tool) for tool in pack.tools]


__all__ = ["OPERATION_KEY", "Operation", "PackError", "Parameter", "Parameters", "PartnerPack", "ResponseShape",
           "SCHEMA_ID", "Tool", "full_tool_name", "pack_schemas", "parse_pack", "placeholders", "split_tool_name",
           "tool_schema"]
