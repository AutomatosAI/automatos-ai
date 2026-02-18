"""
Neural Field — Shared Semantic Space for Agent Coordination
============================================================

PRD-59 Phase 4: Implements the neural field Ψ(x,t), a continuous
semantic vector space shared by multiple agents. Replaces explicit
message-passing with implicit field dynamics.

Components:
  FieldStore   — Redis + pgvector storage for the field
  FieldReader  — Semantic retrieval from other agents' contributions
  FieldWriter  — Write contributions with diffusion dynamics
  Attractor    — Attractor dynamics for agent learning
  Coherence    — Field coherence metric (do agents agree?)

Field evolution equation:
  ∂Ψ/∂t = -∇V(Ψ) + D∇²Ψ + Σᵢ δ(x-xᵢ)Aᵢ

Where:
  Ψ(x,t) ∈ ℝⁿ — field state in embedding space
  V(Ψ)         — task potential (objective function gradient)
  D             — diffusion coefficient (knowledge sharing rate)
  Aᵢ           — agent i's contribution at position xᵢ
"""

from .field_store import NeuralFieldStore, FieldContribution
from .field_reader import FieldReader
from .field_writer import FieldWriter
from .attractor import AttractorLandscape
from .coherence import FieldCoherence

__all__ = [
    "NeuralFieldStore",
    "FieldContribution",
    "FieldReader",
    "FieldWriter",
    "AttractorLandscape",
    "FieldCoherence",
]
