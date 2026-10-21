"""Audited OpenAI calls with durable budgets and explicit render provenance."""
from __future__ import annotations

import base64
import fcntl
import json
import os
from pathlib import Path
import re
import time

from .protocol import digest, file_hash, read_json, write_json, validate_score


class APIBudgetExhausted(RuntimeError):
    pass


class SemanticUnavailable(RuntimeError):
    pass


class APITechnicalFailure(SemanticUnavailable):
    """A transport retry limit or provider configuration error is terminal."""


class IncompleteResponse(RuntimeError):
    pass


def safe_error(exc):
    text = str(exc)
    for name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "<redacted>")
    return type(exc).__name__ + ": " + re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", text)[:2000]


def parse_json(text):
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    value = json.loads(text, parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("API output must be a JSON object")
    return value


class SceneAPI:
    def __init__(self, *, run_dir, image_path, fingerprint, model="gpt-5.6-sol",
                 reasoning_effort="medium", budget=0, client=None,
                 original_layout_path=None):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.image_path = Path(image_path).resolve()
        self.original_hash = file_hash(self.image_path)
        self.fingerprint, self.budget = fingerprint, int(budget)
        if type(budget) is not int or budget < 0:
            raise ValueError('API budget must be nonnegative; zero means explicitly uncapped')
        self.config = {"model": model, "reasoning_effort": reasoning_effort,
                       "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")}
        self.client = client
        self.original_layout_path = original_layout_path
        self.allowed_images = {}
        self.derived_images = {}
        self.ledger = self.run_dir / "api_attempts.jsonl"

    @property
    def attempts(self):
        return len(self.ledger.read_text().splitlines()) if self.ledger.exists() else 0

    def ensure_configured(self):
        if self.client is None and not os.environ.get("OPENAI_API_KEY", "").strip():
            raise SemanticUnavailable("Missing OPENAI_API_KEY; no semantic samples have been dispatched")

    def register_images(self, images):
        for item in images:
            path = Path(item["path"]).resolve()
            # Workers may only register evidence beneath this attempt's directory.
            if not path.is_relative_to(self.run_dir):
                raise ValueError("Candidate image is outside the current attempt")
            actual = file_hash(path)
            if item.get("sha256") != actual:
                raise ValueError("Worker image hash mismatch")
            self.allowed_images[str(path)] = actual

    def register_support_diagram(self, item, catalog):
        path = Path(item['path']).resolve()
        derivation = item['derivation']
        if (not path.is_relative_to(self.run_dir) or file_hash(path) != item['sha256']
                or derivation.get('kind') != 'support_layer_diagram'
                or derivation.get('catalog_sha256') != digest(catalog)):
            raise ValueError('Support diagram does not match this attempt and catalog')
        self.allowed_images[str(path)] = item['sha256']
        self.derived_images[str(path)] = derivation

    def _content(self, prompt, payload, images):
        content = [{"type": "input_text", "text": prompt},
                   {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        audit = []
        for path in [self.image_path, *map(Path, images)]:
            path = path.resolve()
            expected = self.original_hash if path == self.image_path else self.allowed_images.get(str(path))
            if expected is None or file_hash(path) != expected:
                raise ValueError("Only the unchanged input RGB and hash-bound worker images are allowed")
            from PIL import Image
            with Image.open(path) as im:
                fmt = im.format.lower()
            mime = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}.get(fmt)
            if not mime:
                raise ValueError("Unsupported evidence image format")
            audit.append({"path": str(path), "sha256": expected,
                          "source": "original_rgb" if path == self.image_path else "isaac_candidate_render"})
            if str(path) in self.derived_images:
                derivation = self.derived_images[str(path)]
                source = 'support_layer_diagram' if derivation.get('kind') == 'support_layer_diagram' else 'derived_object_comparison'
                audit[-1].update(source=source,derivation=derivation)
            content.append({"type": "input_image", "detail": "high",
                            "image_url": "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()})
        return content, audit

    def _reserve(self, request_name, request_hash, attempt):
        with (self.run_dir / ".api_budget.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.budget and self.attempts >= self.budget:
                raise APIBudgetExhausted(f"API attempt budget exhausted ({self.budget})")
            with self.ledger.open("a") as stream:
                stream.write(json.dumps({"request_name": request_name, "request_sha256": request_hash,
                                         "attempt": attempt, "started_unix": time.time()}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def call(self, request_name, prompt, payload, images=(), *, max_attempts=5):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", request_name):
            raise ValueError("Invalid API request name")
        content, audit = self._content(prompt, payload, images)
        from urdf_core.prompt_cache import prepare
        content, cache_options, cache_receipt = prepare(
            self.config["model"], content, boundaries=sorted({0, 2, len(content) - 1}))
        receipt = {"fingerprint": self.fingerprint, "config": self.config,
                   "prompt": prompt, "payload": payload, "images": audit, "ground_truth_allowed": False,
                   "prompt_cache": cache_receipt}
        request_hash = digest(receipt)
        folder = self.run_dir / "api_calls" / request_name
        folder.mkdir(parents=True, exist_ok=True)
        completed = folder / "completed.json"
        if completed.exists():
            record = read_json(completed)
            if record["request_sha256"] != request_hash or record["response_sha256"] != file_hash(folder / "response.txt"):
                raise RuntimeError("API cache content changed; use a new run")
            if record.get("content_rejection"):
                raise SemanticUnavailable("Cached API content rejected: " + record["content_rejection"])
            try:
                return parse_json((folder / "response.txt").read_text())
            except ValueError as exc:
                raise SemanticUnavailable("Cached API response is not valid JSON") from exc
        if (folder / "request.json").exists() and read_json(folder / "request.json")["request_sha256"] != request_hash:
            raise RuntimeError("Interrupted API request inputs changed")
        write_json(folder / "request.json", {**receipt, "request_sha256": request_hash})
        used = [json.loads(line) for line in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
        used = sum(item["request_name"] == request_name for item in used)
        if used >= max_attempts:
            raise APITechnicalFailure("Durable retry budget exhausted for " + request_name)
        if self.client is None:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise SemanticUnavailable("Missing OPENAI_API_KEY")
            self.client = OpenAI(api_key=key, base_url=self.config["base_url"], timeout=600, max_retries=0)
        for attempt in range(used + 1, max_attempts + 1):
            start = time.monotonic()
            try:
                from .worker_pool import api_slot
                with api_slot():
                    self._reserve(request_name, request_hash, attempt)
                    with self.client.responses.stream(model=self.config["model"],
                            reasoning={"effort": self.config["reasoning_effort"]},
                            input=[{"role": "user", "content": content}], max_output_tokens=8192,
                            **cache_options) as stream:
                        for _ in stream:
                            pass
                        response = stream.get_final_response()
                status = getattr(response, "status", "completed")
                details = getattr(response, "incomplete_details", None)
                reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)
                rejection = reason if status == "incomplete" and reason in {"max_output_tokens", "content_filter"} else None
                if status != "completed" and rejection is None:
                    raise IncompleteResponse("Provider returned an unexplained incomplete response")
                raw = response.output_text
                if not isinstance(raw, str):
                    raw = ""
                if not raw.strip() and rejection is None:
                    rejection = "empty_content"
                (folder / "response.txt").write_text(raw, encoding="utf-8")
                write_json(completed, {"request_sha256": request_hash,
                    "response_sha256": file_hash(folder / "response.txt"),
                    "response_id": response.id, "prompt_cache": cache_receipt,
                    "elapsed_seconds": time.monotonic() - start,
                    "provider_status": status, "content_rejection": rejection})
                if rejection:
                    raise SemanticUnavailable("API content rejected: " + rejection)
                return parse_json(raw)
            except Exception as exc:
                if isinstance(exc, (APIBudgetExhausted, SemanticUnavailable)):
                    raise
                write_json(folder / f"error_{attempt}.json", {"error": safe_error(exc),
                            "elapsed_seconds": time.monotonic() - start})
                from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
                retryable = isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, IncompleteResponse))
                if not retryable or attempt == max_attempts:
                    if retryable or not isinstance(exc, ValueError):
                        raise APITechnicalFailure(safe_error(exc)) from exc
                    raise SemanticUnavailable(safe_error(exc)) from exc
                time.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticUnavailable("API did not return a response")

    def score(self, candidate, sample_index, prompt):
        images = candidate.get("images", candidate.get("physics", {}).get("images", []))
        if len(images) != 2:
            raise SemanticUnavailable("Semantic scoring needs both hash-bound canonical renders")
        self.register_images(images)
        layout = read_json(candidate["layout_path"])
        catalog = [{"object_id": obj["object_id"], "name": obj.get("name", obj["object_id"]),
                    "is_environment_support": bool(obj.get("is_environment_support", False))}
                   for obj in layout["objects_data"]]
        paths=[item['path'] for item in images]
        payload={'object_catalog':catalog}
        from .semantic_crops import comparisons
        from .semantic_observation import POLICY_VERSION
        derived=comparisons(candidate,self.image_path,self.run_dir/'derived_evidence',
                            original_layout_path=self.original_layout_path)
        self.register_images([derived['image']])
        self.derived_images[derived['image']['path']]=derived['binding']
        paths.append(derived['image']['path'])
        payload.update(semantic_policy=POLICY_VERSION, crop_rules=derived['binding']['rules'],
                       image_order=['original_rgb','candidate_fixed_view_1','candidate_fixed_view_2','object_comparison_crops'])
        value = self.call(f"score_{candidate['candidate_id']}_{sample_index}", prompt,
                          payload, paths, max_attempts=5)
        try:
            from .semantic_observation import validate_observation_review
            return validate_observation_review(value, catalog)
        except ValueError as exc:
            raise SemanticUnavailable("Semantic response violates the fixed rubric: " + str(exc)) from exc
