"""Settings: defaults, overridden by .env and environment variables (`ANALOG_<FIELD>`).

Secrets keep their conventional names and are read only from the environment:
`DATABASE_URL`, `EIS_TOKEN` (ЕИС getDocsIP token, optional — КТРУ can be loaded from a file),
`OPENROUTER_API_KEY` (the choice model and embeddings asked live; without it, their kept answers only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid",
        env_prefix="ANALOG_",
        env_file=PROJECT_ROOT / ".env",
        env_ignore_empty=True,
    )

    database_url: str = Field(
        default="postgresql://postgres:analog@localhost:5434/analog",
        validation_alias=AliasChoices("database_url", "DATABASE_URL"),
    )
    # Downloaded source files: <sources_dir>/<source>/<version>/…, outside git
    sources_dir: Path = PROJECT_ROOT / "data" / "sources"
    # Specialist's knowledge (categories.yaml, attributes.yaml) and the procurement history
    knowledge_dir: Path = PROJECT_ROOT / "research"
    positions_file: Path = PROJECT_ROOT / "data" / "positions.xlsx"
    # A specialist's verdicts on likely queries, to show golden cards on a fresh base
    demo_decisions: Path = PROJECT_ROOT / "demo" / "decisions.json"
    # Evaluation queries: a draft of the knowledge is checked against them before it is published
    gold_queries: Path = PROJECT_ROOT / "evaluation" / "gold_queries.json"
    # Pause between requests to public sites, seconds
    request_delay_s: float = Field(default=1.0, ge=0)
    timeout_s: float = Field(default=120.0, gt=0)
    # What the rules leave unread, read by the choice model (ARCHITECTURE.md §3): off — not at all;
    # cache — answers kept in llm_cache only, no key needed; live — ask the model what is not kept;
    # auto — live when OPENROUTER_API_KEY is set, cache otherwise
    llm: Literal["off", "cache", "live", "auto"] = "auto"
    llm_model: str = "typesafe/jev-1.13"
    llm_cache: Path = PROJECT_ROOT / "data" / "llm" / "readings.jsonl"
    llm_timeout_s: float = Field(default=10.0, gt=0)  # a query waits for the model at most this long
    llm_workers: int = Field(default=8, ge=1)  # parallel requests while parsing the history
    # Category candidates by meaning, in the same mode as the model; via OpenRouter in the prototype,
    # a local embedding model of the same name in production
    embedding_model: str = "baai/bge-m3"
    embedding_cache: Path = PROJECT_ROOT / "data" / "llm" / "embeddings.jsonl"
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("openrouter_api_key", "OPENROUTER_API_KEY")
    )

    @property
    def llm_mode(self) -> Literal["off", "cache", "live"]:
        """The mode in effect: auto resolved by whether a key is set."""
        if self.llm != "auto":
            return self.llm
        return "live" if self.openrouter_api_key and self.openrouter_api_key.get_secret_value() else "cache"
