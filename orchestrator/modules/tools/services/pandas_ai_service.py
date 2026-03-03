import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_pandasai_service: Optional["PandasAIService"] = None


def get_pandasai_service() -> Optional["PandasAIService"]:
    """
    Lazy initializer for PandasAI integration.
    Returns None when PandasAI is not installed or not configured.
    """
    global _pandasai_service
    if _pandasai_service is None:
        try:
            _pandasai_service = PandasAIService()
        except Exception as exc:
            logger.warning(f"PandasAI integration disabled: {exc}")
            _pandasai_service = None
    return _pandasai_service if _pandasai_service and _pandasai_service.enabled else None


class PandasAIService:
    """
    Thin wrapper around the PandasAI library to generate natural-language insights
    (and charts) for tabular data returned by SQL queries.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._init_pandasai()

    def _init_pandasai(self) -> None:
        try:
            import pandasai  # type: ignore
            from pandasai.config import Config  # type: ignore
            from pandasai import SmartDataframe  # type: ignore
            from pandasai_litellm.litellm import LiteLLM  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pandasai and pandasai-litellm packages are required") from exc

        from config import config
        api_key = config.PANDASAI_API_KEY or config.OPENAI_API_KEY
        if not api_key:
            raise RuntimeError("No API key found for PandasAI (set PANDASAI_API_KEY or OPENAI_API_KEY)")

        model = config.PANDASAI_MODEL or config.LLM_MODEL or "gpt-4o-mini"
        output_dir = Path(config.PANDASAI_OUTPUT_DIR or "/tmp/pandasai_charts")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Additional directories PandasAI may emit assets into (e.g. exports/charts)
        default_exports = Path.cwd() / "exports" / "charts"
        self._watch_paths = {output_dir, default_exports}

        self._SmartDataframe = SmartDataframe
        self._Config = Config
        self._LiteLLM = LiteLLM
        self._pandasai_module = pandasai
        self._output_dir = output_dir
        self._llm = LiteLLM(model=model, api_key=api_key)
        self._config = Config(
            llm=self._llm,
            save_charts=True,
            charts_output_path=str(output_dir),
        )
        self.enabled = True
        logger.info("✅ PandasAI service initialized (model=%s, output=%s)", model, output_dir)

    def generate_insight(
        self,
        question: str,
        rows: Sequence[Dict[str, Any]],
        columns: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Produce a natural-language insight (and optional chart assets) for the provided data.
        Returns None if PandasAI is disabled or an error occurred.
        """
        if not self.enabled:
            return None
        if not rows:
            return None

        try:
            df = pd.DataFrame(list(rows))
            if columns:
                df = df.loc[:, list(columns)]

            # Normalize types to help PandasAI/PDuckDB reason about the data
            for col in df.columns:
                series = df[col]
                if pd.api.types.is_object_dtype(series):
                    # Attempt datetime coercion first; if it fails the original series is returned
                    coerced = pd.to_datetime(series, errors="ignore", utc=False)
                    if not coerced.equals(series):
                        df[col] = coerced
                        continue

                    numeric_coerced = pd.to_numeric(series, errors="ignore")
                    if not numeric_coerced.equals(series):
                        df[col] = numeric_coerced

            # Force any unsigned / nullable integer dtypes into plain int64 for DuckDB compatibility
            for col in df.columns:
                series = df[col]
                dtype = series.dtype
                kind = getattr(dtype, "kind", "")
                if kind == "u" or (pd.api.types.is_integer_dtype(series) and dtype != "int64"):
                    df[col] = series.astype("int64")
        except Exception as exc:
            logger.error("PandasAI: failed to build DataFrame: %s", exc, exc_info=True)
            return None

        baseline_files = self._snapshot_outputs()

        requested_plot = False
        lowered_question = question.lower()
        if any(keyword in lowered_question for keyword in ("chart", "plot", "graph", "visual")):
            requested_plot = True
        elif len(df.select_dtypes(include=["number"]).columns) >= 1:
            # Auto-request a visualization when numeric measures exist
            requested_plot = True

        try:
            smart_df = self._SmartDataframe(df, config=self._config)
            enriched_question = question
            if requested_plot:
                extra_guidance = (
                    "Please create a high-quality matplotlib or seaborn chart with vibrant colors, "
                    "clear labels, and an informative title. Save the chart as a PNG file in the "
                    "configured output directory and return the file path so it can be displayed."
                )

                if {"metric_name", "metric"}.intersection(df.columns):
                    extra_guidance += (
                        " If the dataset contains a metric/category column (e.g., metric_name), plot separate "
                        "series on the same chart with distinct colors and include a legend."
                    )

                if {"recorded_at", "timestamp", "created_at"}.intersection(df.columns):
                    extra_guidance += (
                        " Ensure the x-axis properly reflects chronological order and formats timestamps for readability."
                    )

                enriched_question = f"{question}\n\n{extra_guidance}"

            summary = smart_df.chat(enriched_question)
        except Exception as exc:
            logger.error("PandasAI execution failed: %s", exc, exc_info=True)
            return {
                "summary": f"PandasAI could not generate an insight: {exc}",
                "charts": [],
                "error": str(exc),
            }

        new_files = self._snapshot_outputs() - baseline_files
        charts: List[Dict[str, Any]] = []

        if requested_plot and isinstance(summary, str):
            generated_path = self._resolve_generated_path(summary)
            if generated_path:
                new_files.add(generated_path)
                summary = ""
            else:
                logger.warning(
                    "PandasAI reported plot output but file not found: %s", summary
                )

        if isinstance(summary, str):
            # PandasAI may return a file path even when plot wasn't explicitly requested
            possible_file = self._resolve_generated_path(summary)
            if possible_file:
                new_files.add(possible_file)
                summary = ""

        for path in new_files:
            try:
                data = path.read_bytes()
                charts.append(
                    {
                        "filename": path.name,
                        "mime_type": self._guess_mime(path),
                        "base64": base64.b64encode(data).decode("utf-8"),
                    }
                )
            except Exception as exc:
                logger.warning("PandasAI: failed to read chart %s: %s", path, exc)

        return {
            "summary": str(summary) if summary is not None else "",
            "charts": charts,
        }

    def _snapshot_outputs(self) -> set:
        files = set()
        for watch_path in self._watch_paths:
            files.update({path for path in watch_path.glob("**/*") if path.is_file()})
        return files

    def _resolve_generated_path(self, output: str) -> Optional[Path]:
        candidate = Path(output)
        candidates = [candidate]

        if not candidate.is_absolute():
            candidates.append(Path.cwd() / candidate)
            candidates.extend(path / candidate.name for path in self._watch_paths)

        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    @staticmethod
    def _guess_mime(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".png"}:
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix in {".gif"}:
            return "image/gif"
        return "application/octet-stream"

