"""Resumable summarization generation and independently cached quality metrics.

Model clients and scorers are supplied by the caller. Importing this module does
not initialize models, contact providers, or configure MLflow.
"""

import hashlib
import json
import math
import os
import statistics
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter
from uuid import uuid4


class GenerationResponseError(ValueError):
    """Keep the response available when no usable final summary was returned."""

    def __init__(self, message: str, metadata: dict):
        super().__init__(message)
        self.metadata = metadata


def fingerprint(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_configs(path: Path, *, environ=None) -> list[dict]:
    """Read shared defaults and enabled model entries without creating clients.

    Credentials stay in the environment. Each returned configuration is complete
    so it can be frozen into a generation manifest.
    """
    environment = os.environ if environ is None else environ
    document = read_json(path)
    if not isinstance(document, dict) or set(document) - {"defaults", "models"}:
        raise ValueError("Model config file must contain 'defaults' and 'models' only.")
    defaults = document.get("defaults", {})
    entries = document.get("models")
    if not isinstance(defaults, dict) or not isinstance(entries, list) or not entries:
        raise ValueError("Expected a defaults object and a nonempty models list.")
    common_keys = {"temperature", "max_output_tokens", "timeout_seconds"}
    if set(defaults) - common_keys:
        raise ValueError(f"Unsupported shared defaults: {sorted(set(defaults) - common_keys)}")
    configurations = []
    for index, entry in enumerate(entries, start=1):
        label = f"Model entry {index}"
        if not isinstance(entry, dict) or not isinstance(entry.get("enabled", True), bool):
            raise ValueError(f"{label}: expected an object with optional boolean enabled.")
        if not entry.get("enabled", True):
            continue
        config = {"temperature": 0, "max_output_tokens": 2048, "timeout_seconds": 300,
                  **defaults, **entry}
        config.pop("enabled", None)
        provider = config.get("provider")
        if provider not in {"ollama", "openrouter"}:
            raise ValueError(f"{label}: provider must be ollama or openrouter.")
        allowed = common_keys | {"provider", "model", "tokenizer", "tokenizer_revision"}
        if provider == "ollama":
            allowed |= {"base_url", "num_ctx", "num_gpu", "reasoning", "top_k",
                        "repeat_penalty", "tokenizer", "tokenizer_revision"}
            config = {
                "base_url": environment.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
                "num_ctx": int(environment.get("OLLAMA_NUM_CTX", "32768")),
                "num_gpu": 0, "reasoning": False, "top_k": 50, "repeat_penalty": 1.1,
                "tokenizer_revision": "main", **config,
            }
        if set(config) - allowed:
            raise ValueError(f"{label}: unsupported settings {sorted(set(config) - allowed)}")
        for key in (["model", "base_url", "tokenizer", "tokenizer_revision"]
                    if provider == "ollama" else ["model"]):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"{label}: {key} must be a nonempty string.")
        for key in ["tokenizer", "tokenizer_revision"]:
            if key in config and (not isinstance(config[key], str) or not config[key].strip()):
                raise ValueError(f"{label}: {key} must be a nonempty string.")
        for key in ["temperature", "timeout_seconds"]:
            value = config[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (key == "timeout_seconds" and value == 0)):
                raise ValueError(f"{label}: invalid {key}.")
        for key in (["max_output_tokens", "num_ctx", "top_k"]
                    if provider == "ollama" else ["max_output_tokens"]):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f"{label}: {key} must be a positive integer.")
        if provider == "ollama":
            if config["num_ctx"] <= config["max_output_tokens"]:
                raise ValueError(f"{label}: num_ctx must leave space for input.")
            if type(config["num_gpu"]) is not int or config["num_gpu"] < -1:
                raise ValueError(f"{label}: num_gpu must be an integer >= -1.")
            if not isinstance(config["reasoning"], bool):
                raise ValueError(f"{label}: reasoning must be a boolean.")
            penalty = config["repeat_penalty"]
            if (isinstance(penalty, bool) or not isinstance(penalty, (int, float))
                    or not math.isfinite(penalty) or penalty <= 0):
                raise ValueError(f"{label}: repeat_penalty must be positive.")
        configurations.append(config)
    if not configurations:
        raise ValueError("Enable at least one model in the config file.")
    return configurations


def write_json(path: Path, value) -> None:
    path = Path(path)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_generation(directory: Path, manifest: dict) -> Path:
    """Create a frozen manifest, or verify it exactly when resuming.

    Use a different directory for each new repetition. A directory is owned by
    one notebook/kernel at a time; concurrent writers are not supported.
    """
    directory = Path(directory)
    articles = manifest["articles"]
    ids = [article["id"] for article in articles]
    if not articles or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("Expected articles with nonempty string IDs.")
    if len(set(ids)) != len(ids):
        raise ValueError("Article IDs must be unique.")
    for article in articles:
        if not article["text"].strip() or not article["human_summaries"]:
            raise ValueError("Each article needs text and human references.")
        if any(not reference.strip() for reference in article["human_summaries"]):
            raise ValueError("Human references must be nonempty.")
    # Validate JSON serializability before creating any files.
    fingerprint(manifest)
    path = directory / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Manifest differs; use a new generation directory.")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Refusing to reuse a nonempty directory without a manifest.")
        write_json(path, manifest)
    return directory


def generation_path(directory: Path, article_id: str) -> Path:
    return Path(directory) / "generations" / f"{fingerprint(article_id)}.json"


def generation_records(directory: Path) -> list[dict]:
    """Read checkpoints in dataset order, validating their source identity."""
    manifest = read_json(Path(directory) / "manifest.json")
    records = []
    for article in manifest["articles"]:
        path = generation_path(directory, article["id"])
        if path.exists():
            record = read_json(path)
            if (record["article_id"] != article["id"]
                    or record["article_sha256"] != fingerprint(article)):
                raise ValueError(f"Checkpoint source mismatch: {article['id']}")
            records.append(record)
    return records


def generate_articles(
    directory: Path,
    generate: Callable,
    progress: Callable[[Iterable], Iterable] = iter,
) -> list[dict]:
    """Resume successful articles; retry failed ones only on explicit rerun.

    The callback receives only id/text, never references or human ratings, and
    returns (clean_summary, metadata). Preserve the raw output in metadata.
    """
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    previous = {item["article_id"]: item for item in generation_records(directory)}
    for article in progress(manifest["articles"]):
        if previous.get(article["id"], {}).get("status") == "ok":
            continue
        record = {
            "article_id": article["id"],
            "article_sha256": fingerprint(article),
        }
        started = perf_counter()
        try:
            summary, metadata = generate({"id": article["id"], "text": article["text"]})
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("Model returned an empty summary.")
            record.update(status="ok", summary=summary.strip(), metadata=metadata)
            fingerprint(record)
        except Exception as error:
            record = {
                "article_id": article["id"],
                "article_sha256": fingerprint(article),
                "status": "error",
                "error": traceback.format_exc(),
            }
            if isinstance(error, GenerationResponseError):
                record["metadata"] = error.metadata
        record["generation_seconds"] = perf_counter() - started
        write_json(generation_path(directory, article["id"]), record)
    return generation_records(directory)


def score_path(directory: Path, metric_spec: dict, generation: dict) -> Path:
    return (Path(directory) / "scores" / fingerprint(metric_spec)
            / f"{fingerprint(generation)}.json")


def score_records(directory: Path, metric_spec: dict) -> list[dict]:
    """Only return scores belonging to the current successful generation."""
    records = []
    for generation in generation_records(directory):
        if generation["status"] != "ok":
            continue
        path = score_path(directory, metric_spec, generation)
        if path.exists():
            record = read_json(path)
            if (record["generation_sha256"] != fingerprint(generation)
                    or record["metric_id"] != fingerprint(metric_spec)
                    or record["article_id"] != generation["article_id"]):
                raise ValueError(f"Score checkpoint mismatch: {path}")
            records.append(record)
    return records


def score_articles(
    directory: Path,
    metric_spec: dict,
    evaluate: Callable,
    progress: Callable[[Iterable], Iterable] = iter,
) -> list[dict]:
    """Add one metric without calling the generator or other metrics.

    evaluate(article, summary) returns a float or a dict containing 'value' and
    optional diagnostic fields. A lazy evaluator avoids loading cached models.
    """
    directory = Path(directory)
    metric_id = fingerprint(metric_spec)
    metric_path = directory / "scores" / metric_id / "metric.json"
    if metric_path.exists() and read_json(metric_path) != metric_spec:
        raise ValueError("Metric manifest differs.")
    write_json(metric_path, metric_spec)
    manifest = read_json(directory / "manifest.json")
    articles = {article["id"]: article for article in manifest["articles"]}
    cached = {item["article_id"]: item for item in score_records(directory, metric_spec)}
    generations = [item for item in generation_records(directory) if item["status"] == "ok"]
    for generation in progress(generations):
        article_id = generation["article_id"]
        if cached.get(article_id, {}).get("status") == "ok":
            continue
        record = {
            "article_id": article_id,
            "generation_sha256": fingerprint(generation),
            "metric_id": metric_id,
        }
        started = perf_counter()
        try:
            result = evaluate(articles[article_id], generation["summary"])
            result = result if isinstance(result, dict) else {"value": result}
            value = float(result["value"])
            if not math.isfinite(value):
                raise ValueError("Metric returned a non-finite score.")
            record.update(status="ok", value=value, diagnostics={
                key: value for key, value in result.items() if key != "value"
            })
            fingerprint(record)
        except Exception:
            record = {
                "article_id": article_id,
                "generation_sha256": fingerprint(generation),
                "metric_id": metric_id,
                "status": "error",
                "error": traceback.format_exc(),
            }
        record["scoring_seconds"] = perf_counter() - started
        write_json(score_path(directory, metric_spec, generation), record)
    return score_records(directory, metric_spec)


def publish_results(directory: Path, metric_spec: dict | None = None, *, interrupted=False) -> str:
    """Mirror local results to the currently configured MLflow experiment.

    Reuse the corresponding tracking run on resume. Failed/partial runs retain
    their artifacts and explicit coverage, and never masquerade as complete.
    Local checkpoints remain usable even if publication fails.
    """
    import mlflow

    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    generation = generation_records(directory)
    successful = [row for row in generation if row["status"] == "ok"]
    expected = len(manifest["articles"])
    kind = "evaluation" if metric_spec is not None else "generation"
    tracking_path = directory / "tracking.json"
    tracking = read_json(tracking_path) if tracking_path.exists() else {}
    experiment = mlflow.get_experiment_by_name(manifest["experiment_name"])
    if experiment is None:
        experiment_id = mlflow.create_experiment(manifest["experiment_name"])
    else:
        experiment_id = experiment.experiment_id
    tracking_key = fingerprint([mlflow.get_tracking_uri(), experiment_id])
    links = tracking.setdefault(tracking_key, {})
    metric_id = fingerprint(metric_spec) if metric_spec is not None else None
    run_key = metric_id or "generation"
    tags = {
        "run_type": kind,
        "benchmark_group": manifest["benchmark_group"],
        "batch_id": manifest["batch_id"],
        "generation_id": manifest["generation_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "config_sha256": fingerprint(manifest["generator"]),
    }
    if metric_spec is not None:
        tags.update(metric_id=metric_id, metric_name=metric_spec["name"])
        if "generation" in links:
            tags["generation_run_id"] = links["generation"]

    with mlflow.start_run(
        run_id=links.get(run_key), experiment_id=experiment_id,
        run_name=f"{kind}: {manifest['generator']['model']} ({directory.name[:8]})",
    ) as run:
        links[run_key] = run.info.run_id
        write_json(tracking_path, tracking)
        mlflow.set_tags(tags)
        mlflow.log_params({
            "provider": manifest["generator"]["provider"],
            "model": manifest["generator"]["model"],
            "repeat_index": manifest["repeat_index"],
            "expected_articles": expected,
            "manifest_sha256": fingerprint(manifest),
        })
        mlflow.log_dict(manifest, "config/manifest.json")
        metrics = {
            "expected_articles": expected,
            "generated_articles": len(successful),
            "generation_errors": sum(row["status"] == "error" for row in generation),
            "missing_generations": expected - len(generation),
            "truncated_articles": sum(
                row["metadata"].get("finish_reason") in {"length", "max_tokens"}
                for row in successful
            ),
        }
        if metric_spec is None:
            completed = len(successful)
            metrics["generation_seconds"] = sum(row["generation_seconds"] for row in generation)
            metrics["request_seconds"] = sum(row["metadata"].get("request_seconds", 0) for row in successful)
            if successful:
                metrics["summary_words_mean"] = statistics.mean(
                    len(row["summary"].split()) for row in successful
                )
            for token_kind in ("input_tokens", "output_tokens"):
                counts = [row["metadata"].get("usage", {}).get(token_kind) for row in successful]
                known = [count for count in counts if count is not None]
                metrics[f"{token_kind}_reported_articles"] = len(known)
                if known:
                    metrics[token_kind] = sum(known)
            mlflow.log_dict({"articles": generation}, "outputs/generations.json")
            setup_error = directory / "setup_error.json"
            if setup_error.exists():
                mlflow.log_dict(read_json(setup_error), "errors/generation_setup.json")
        else:
            scores = score_records(directory, metric_spec)
            values = [row["value"] for row in scores if row["status"] == "ok"]
            completed = len(values)
            metrics.update(
                scored_articles=completed,
                scoring_errors=sum(row["status"] == "error" for row in scores),
                missing_scores=len(successful) - len(scores),
                scoring_seconds=sum(row["scoring_seconds"] for row in scores),
            )
            if values:
                metrics.update(score_mean=statistics.mean(values), score_median=statistics.median(values))
            mlflow.log_dict(metric_spec, "config/metric.json")
            mlflow.log_dict({"articles": scores}, "metrics/per_article.json")
            setup_error = directory / "scores" / metric_id / "setup_error.json"
            if setup_error.exists():
                mlflow.log_dict(read_json(setup_error), "errors/scorer_setup.json")
        metrics["coverage"] = completed / expected
        mlflow.log_metrics(metrics)
        status = "KILLED" if interrupted else ("FINISHED" if completed == expected else "FAILED")
        mlflow.set_tag("result_status", status)
    mlflow.MlflowClient().set_terminated(run.info.run_id, status=status)
    return run.info.run_id


def compare_generations(directories: list[Path], metric_spec: dict) -> tuple[list[dict], list[str]]:
    """Compare only matching protocols and paired article IDs for one metric.

    Return one row per repetition, plus the exact paired article IDs. Repeated
    generations are separate observations, not extra independent articles.
    """
    if not directories:
        return [], []
    manifests = [read_json(Path(path) / "manifest.json") for path in directories]
    if len({manifest["protocol_sha256"] for manifest in manifests}) != 1:
        raise ValueError("Comparison requires the same dataset and prompt protocol.")
    scores = [
        {row["article_id"]: row["value"] for row in score_records(path, metric_spec)
         if row["status"] == "ok"}
        for path in directories
    ]
    common = sorted(set.intersection(*(set(group) for group in scores)))
    rows = []
    for directory, manifest, group in zip(directories, manifests, scores):
        values = list(group.values())
        paired = [group[article_id] for article_id in common]
        expected = len(manifest["articles"])
        rows.append({
            "generation_id": manifest["generation_id"],
            "provider": manifest["generator"]["provider"],
            "model": manifest["generator"]["model"],
            "config_sha256": fingerprint(manifest["generator"]),
            "repeat_index": manifest["repeat_index"],
            "metric": metric_spec["name"],
            "expected_articles": expected,
            "scored_articles": len(values),
            "coverage": len(values) / expected,
            "paired_articles": len(paired),
            "excluded_articles": expected - len(paired),
            "score_mean": statistics.mean(values) if values else None,
            "score_median": statistics.median(values) if values else None,
            "paired_mean": statistics.mean(paired) if paired else None,
            "paired_median": statistics.median(paired) if paired else None,
            "directory": str(directory),
        })
    return rows, common
