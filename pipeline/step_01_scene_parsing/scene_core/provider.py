"""Bounded provider calls with replayable, credential-free input receipts.

This module accepts only the original RGB image, pipeline-produced previews,
generic prompts, and the current run's model/validation output. It never reads GT.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "step_02_object_generation"))
from urdf_core.prompt_cache import prepare as prepare_cache

TEXT_SOURCES = {"prompt", "inventory", "layout", "critic", "diagnostics", "validation_errors", "perception", "orientation"}


def registered_perception_image(path: Path, run_dir: Path, original_sha256: str, fingerprint: str) -> bool:
    """Images must be explicitly registered by this RGB-only perception run."""
    path, root = path.resolve(), run_dir.resolve()
    manifest = root / "perception_input_manifest.json"
    if not path.is_relative_to(root) or not manifest.is_file():
        return False
    relative = str(path.relative_to(root))
    if not (relative.startswith(("perception/", "orientation/")) or re.fullmatch(
            r"(?:localization|layout_reprojection)_round_\d+\.png", relative)):
        return False
    try:
        data = json.loads(manifest.read_text())
        record = data["images"][relative]
        return (data.get("schema_version") == "perception_inputs.v1" and
                data.get("original_sha256") == original_sha256 and data.get("fingerprint") == fingerprint and
                record.get("source") == ("orientation_derived" if relative.startswith("orientation/") else "rgb_derived_perception") and path.is_file() and
                record.get("sha256") == digest_bytes(path.read_bytes()))
    except (ValueError, KeyError, OSError, TypeError):
        return False


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def provider_config(model: str, reasoning_effort: str = "high", thinking_budget: int = 8192) -> dict:
    if model.startswith("gpt-"):
        return {"provider": "openai", "model": model,
                "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
                "key_env": "OPENAI_API_KEY", "reasoning_effort": reasoning_effort,
                "max_output_tokens": 16384}
    if model.startswith("qwen"):
        return {"provider": "dashscope", "model": model,
                "base_url": os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/"),
                "key_env": "DASHSCOPE_API_KEY", "enable_thinking": True,
                "thinking_budget": thinking_budget, "max_output_tokens": 16384}
    raise ValueError(f"Unsupported scene model {model!r}; explicitly configure a supported provider/model.")


def safe_error(exc: Exception, secret: str = "") -> str:
    message = str(exc)
    if secret:
        message = message.replace(secret, "<redacted>")
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", message)
    return f"{type(exc).__name__}: {message[:2000]}"


def recorded_api_attempts(run_dir: Path) -> int:
    """Count durable attempt reservations, including interrupted/failed requests."""
    return sum(len(path.read_text().splitlines())
               for path in (run_dir / 'api_calls').glob('*/attempts.jsonl'))


class ProviderResponseError(RuntimeError):
    """The provider ended a stream without a complete response."""


class ModelAPI:
    def __init__(self, config: dict, image_path: Path, run_dir: Path, fingerprint: str, *, max_attempts: int = 5):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
            raise ValueError('max_attempts must be an integer between 1 and 5')
        if 'max_run_api_attempts' in config or 'max_run_seconds' in config:
            raise ValueError('Run-wide parsing limits are retired; use the content refinement and per-request attempt budgets')
        self.max_attempts = max_attempts
        self.config = config
        self.image_path = image_path.resolve()
        self.run_dir = run_dir.resolve()
        self.fingerprint = fingerprint
        self.original_sha256 = digest_bytes(self.image_path.read_bytes())
        self.secret = os.environ.get(config["key_env"], "").strip()
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.secret:
                raise RuntimeError(f"Missing {self.config['key_env']}; no model substitution is allowed.")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.secret, base_url=self.config["base_url"],
                                  timeout=600.0, max_retries=0)
        return self._client

    def _inputs(self, texts: list[tuple[str, str]], images: list[Path]) -> tuple[list, list]:
        audit, content = [], []
        for source, text in texts:
            if source not in TEXT_SOURCES:
                raise ValueError(f"Unapproved input source: {source}")
            audit.append({"type": "text", "source": source, "text": text,
                          "sha256": digest_bytes(text.encode())})
            content.append({"type": "text", "text": text})
        for path in images:
            path = Path(path).resolve()
            if path == self.image_path:
                source = "original_rgb"
            elif path.parent == self.run_dir and re.fullmatch(
                    r"layout_(?:2d|3d|top|front|side)_round_\d+\.png", path.name):
                source = "generated_layout_visualization"
            elif registered_perception_image(path, self.run_dir, self.original_sha256, self.fingerprint):
                source = "generated_orientation_visualization" if path.is_relative_to(self.run_dir/'orientation') else "generated_perception_visualization"
            else:
                raise ValueError("Only the original RGB and this run's generated layout views are allowed.")
            data = path.read_bytes()
            if source == "original_rgb" and digest_bytes(data) != self.original_sha256:
                raise RuntimeError("Original RGB changed during the run")
            from PIL import Image
            with Image.open(path) as im:
                fmt = (im.format or "").lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(fmt)
            if not mime:
                raise ValueError("Unsupported image format")
            audit.append({"type": "image", "source": source, "path": str(path),
                          "sha256": digest_bytes(data)})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}})
        return audit, content

    def call(self, stage: str, texts: list[tuple[str, str]], images: list[Path]) -> str:
        if not re.fullmatch(r"[a-z0-9_]+", stage):
            raise ValueError("Invalid call stage")
        call_dir = self.run_dir / "api_calls" / stage
        call_dir.mkdir(parents=True, exist_ok=True)
        inputs, content = self._inputs(texts, images)
        cache_options, cache_receipt = {}, None
        if self.config["provider"] == "openai":
            response_content = [
                {"type": "input_text", "text": x["text"]} if x["type"] == "text" else
                {"type": "input_image", "image_url": x["image_url"]["url"], "detail": "high"}
                for x in content]
            fixed = 0
            for source, _ in texts:
                if source != "prompt":
                    break
                fixed += 1
            response_content, cache_options, cache_receipt = prepare_cache(
                self.config["model"], response_content, boundaries=[fixed - 1] if fixed else [])
        receipt = {"stage": stage, "fingerprint": self.fingerprint,
                   "config": self.config, "inputs": inputs, "prompt_cache": cache_receipt}
        request_digest = digest_bytes(json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode())
        receipt["request_sha256"] = request_digest
        completed = call_dir / "completed.json"
        response_path = call_dir / "response.txt"
        if completed.exists():
            cached = json.loads(completed.read_text())
            if cached.get("request_sha256") != request_digest:
                raise RuntimeError("Cached API input changed; use --force_layout for a new run.")
            raw = response_path.read_bytes()
            if digest_bytes(raw) != cached.get("response_sha256"):
                raise RuntimeError("Cached API response was modified; refusing replay.")
            return raw.decode()
        request_path = call_dir / "request.json"
        if request_path.exists() and json.loads(request_path.read_text()) != receipt:
            raise RuntimeError("Interrupted API input changed; use --force_layout for a new run.")
        write_json(request_path, receipt)
        attempts_path = call_dir / "attempts.jsonl"
        # Retry attempts survive process restarts. Resuming cannot reset the budget.
        attempts = len(attempts_path.read_text().splitlines()) if attempts_path.exists() else 0
        if attempts >= self.max_attempts:
            raise RuntimeError(f"API retry budget exhausted at {stage}; start a new run explicitly.")
        for attempt in range(attempts + 1, self.max_attempts + 1):
            client = self._get_client()
            started = time.monotonic()
            with attempts_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"attempt": attempt, "started_unix": time.time(),
                                    "request_sha256": request_digest}) + "\n")
            print(f"[{self.config['model']}] {stage} attempt {attempt}", flush=True)
            try:
                if self.config["provider"] == "openai":
                    with client.responses.stream(model=self.config["model"],
                            input=[{"role": "user", "content": response_content}],
                            reasoning={"effort": self.config["reasoning_effort"]},
                            max_output_tokens=self.config["max_output_tokens"], **cache_options) as stream:
                        for _ in stream:
                            pass
                        result = stream.get_final_response()
                    if getattr(result, "status", "completed") not in ("completed", "incomplete"):
                        raise ProviderResponseError(f"Provider returned incomplete response: {getattr(result, 'status', None)}")
                    output = result.output_text or ''
                    if getattr(result, 'status', 'completed') == 'incomplete':
                        (call_dir / 'incomplete_content.txt').write_text(output, encoding='utf-8')
                        output = ''
                    response_id = result.id
                else:
                    chunks, response_id = [], None
                    stream = client.chat.completions.create(model=self.config["model"],
                        messages=[{"role": "user", "content": content}], stream=True,
                        extra_body={"enable_thinking": True, "thinking_budget": self.config["thinking_budget"]},
                        max_tokens=self.config["max_output_tokens"])
                    finish_reason = None
                    try:
                        for chunk in stream:
                            response_id = chunk.id or response_id
                            for choice in chunk.choices:
                                if choice.delta.content:
                                    chunks.append(choice.delta.content)
                                if choice.finish_reason:
                                    finish_reason = choice.finish_reason
                    finally:
                        stream.close()
                    if finish_reason not in ("stop", "length", "content_filter"):
                        raise ProviderResponseError(f"Provider returned incomplete response: {finish_reason}")
                    output = "".join(chunks) if finish_reason == 'stop' else ''
                response_path.write_text(output, encoding="utf-8")
                write_json(completed, {"request_sha256": request_digest,
                    "response_sha256": digest_bytes(response_path.read_bytes()),
                    "response_id": response_id, "prompt_cache": cache_receipt,
                    "elapsed_seconds": time.monotonic() - started, "attempt": attempt})
                return output
            except Exception as exc:
                error = safe_error(exc, self.secret)
                write_json(call_dir / f"error_{attempt}.json", {"error": error,
                           "elapsed_seconds": time.monotonic() - started, "attempt": attempt})
                from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
                if attempt >= self.max_attempts or not isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, ProviderResponseError)):
                    raise RuntimeError(error) from None
                time.sleep(2 ** attempt)
        raise RuntimeError("Unreachable provider state")


def aggregate_calls(run_dir: Path) -> dict:
    result = {"calls": 0, "attempts": 0, "elapsed_seconds": 0.0, "per_stage": {}}
    counters = tuple(key for key in result if key != 'per_stage')
    for call_dir in sorted((run_dir / 'api_calls').glob('*')):
        if not call_dir.is_dir():
            continue
        stage = {key: 0.0 if key == 'elapsed_seconds' else 0 for key in counters}
        stage.update(image_count=0, text_chars=0)
        completed, request, attempts = (call_dir / name for name in
                                        ('completed.json', 'request.json', 'attempts.jsonl'))
        if not any(path.is_file() for path in (completed, request, attempts)):
            continue
        if completed.is_file():
            data = json.loads(completed.read_text())
            stage['calls'] = 1
            # Preserve the existing successful-request latency metric. Failed or
            # interrupted attempts remain visible in attempts and error receipts.
            stage['elapsed_seconds'] = data.get('elapsed_seconds', 0) or 0
        if attempts.is_file():
            stage['attempts'] = len(attempts.read_text().splitlines())
        if request.is_file():
            inputs = json.loads(request.read_text()).get('inputs', [])
            # Request size describes this logical stage once, not once per retry.
            stage['image_count'] = sum(item.get('type') == 'image' for item in inputs)
            stage['text_chars'] = sum(len(item.get('text', '')) for item in inputs
                                      if item.get('type') == 'text')
        result['per_stage'][call_dir.name] = stage
        for key in counters:
            result[key] += stage[key]
    return result


def audit_inputs(run_dir: Path, image_path: Path, image_sha256: str, fingerprint: str) -> dict:
    """Post-run provenance check, independent of model self-assessment."""
    errors, requests = [], 0
    run_dir, image_path = run_dir.resolve(), image_path.resolve()
    for path in sorted((run_dir / "api_calls").glob("*/request.json")):
        requests += 1
        data = json.loads(path.read_text())
        recorded_digest = data.pop("request_sha256", None)
        if digest_bytes(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()) != recorded_digest:
            errors.append(f"{path.parent.name}: altered request receipt")
        if data.get("fingerprint") != fingerprint:
            errors.append(f"{path.parent.name}: fingerprint mismatch")
        for item in data.get("inputs", []):
            source = item.get("source")
            if item.get("type") == "text":
                if source not in TEXT_SOURCES or digest_bytes(item.get("text", "").encode()) != item.get("sha256"):
                    errors.append(f"{path.parent.name}: invalid text source or digest")
            elif item.get("type") == "image":
                image = Path(item.get("path", "")).resolve()
                if source == "original_rgb":
                    if image != image_path or item.get("sha256") != image_sha256:
                        errors.append(f"{path.parent.name}: unexpected original image")
                elif source == "generated_layout_visualization":
                    if image.parent != run_dir or not re.fullmatch(r"layout_(?:2d|3d|top|front|side)_round_\d+\.png", image.name):
                        errors.append(f"{path.parent.name}: unexpected visualization source")
                elif source in {"generated_perception_visualization", "generated_orientation_visualization"}:
                    if not registered_perception_image(image, run_dir, image_sha256, fingerprint):
                        errors.append(f"{path.parent.name}: unbound perception visualization")
                    elif (source == 'generated_orientation_visualization') != image.is_relative_to(run_dir/'orientation'):
                        errors.append(f"{path.parent.name}: incorrect derived image classification")
                else:
                    errors.append(f"{path.parent.name}: disallowed image source")
                if not image.is_file() or digest_bytes(image.read_bytes()) != item.get("sha256"):
                    errors.append(f"{path.parent.name}: image digest mismatch")
            else:
                errors.append(f"{path.parent.name}: disallowed input type")
    return {"passed": not errors, "requests_checked": requests,
            "allowed_sources": sorted(TEXT_SOURCES | {"original_rgb", "generated_layout_visualization", "generated_perception_visualization", "generated_orientation_visualization"}),
            "ground_truth_allowed": False, "errors": errors}
