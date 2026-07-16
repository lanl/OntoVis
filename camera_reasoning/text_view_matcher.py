"""
Text-embedding based reference-view matcher — the text-side counterpart to
image_view_matcher.py. Compares a query text (e.g. the LLM's own "visual
observation" of the current screenshot) against the existing per-view text
descriptions (reference_views/view_descriptions.json) using CLIP's text encoder
(same model family already used for image embeddings, so no new dependency),
cosine similarity on normalized embeddings.

Self-contained (no relative imports), so it also runs fine as a standalone script.

CLI usage:
  python camera_reasoning/text_view_matcher.py \\
      --query-text "Top view showing the dorsal surface of the foot." \\
      --descriptions reference_views/view_descriptions.json \\
      --top-k 5 \\
      --cache reference_text_embeddings.pkl
"""
import argparse
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

DEFAULT_DESCRIPTIONS_PATH = "reference_views/view_descriptions.json"
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

# confidence_margin below this is flagged ambiguous (best/second-best views are too
# close to be confident) — same threshold convention as image_view_matcher.py.
AMBIGUITY_THRESHOLD = 0.02


def _resolve_device(device: Optional[str]) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return vector
    return vector / norm


@dataclass
class ReferenceTextEmbedding:
    view_name: str
    text: str
    embedding: np.ndarray  # L2-normalized, shape (D,)


class TextViewMatcher:
    """Matches a query text against a bank of reference-view text descriptions via
    CLIP text embeddings and cosine similarity.

    Typical usage:
        matcher = TextViewMatcher("reference_views/view_descriptions.json",
                                   cache_path="reference_text_embeddings.pkl")
        matcher.build_index()
        result = matcher.match("Top view showing the dorsal surface.", top_k=5)
    """

    def __init__(
        self,
        descriptions_path: str = DEFAULT_DESCRIPTIONS_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.descriptions_path = Path(descriptions_path)
        if not self.descriptions_path.exists():
            raise FileNotFoundError(
                f"View description file not found: {self.descriptions_path}. "
                "Generate descriptions first (see camera_reasoning/view_description_generator.py), "
                "or pass the correct --descriptions path."
            )

        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = _resolve_device(device)

        self._model = None
        self._processor = None
        self.index: List[ReferenceTextEmbedding] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazily load the CLIP model — avoids paying the (slow) model-load cost
        for callers that only want to load a cache and never encode anything.
        """
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        print(f"[text_view_matcher] Loading {self.model_name!r} on {self.device}...")
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    # ------------------------------------------------------------------
    # Reference discovery
    # ------------------------------------------------------------------

    def _discover_reference_texts(self) -> List[dict]:
        """Return the view_descriptions.json entries as (label, text) pairs,
        skipping any that don't match a known schema rather than crashing.

        Supports both:
          - the semantic schema:        {"view_name": ..., "short_description": ...}
          - the neutral graph schema:   {"node_id": ..., "description": ...}
        (see view_description_generator.py vs. graph_view_description_generator.py)
        """
        entries = json.loads(self.descriptions_path.read_text())
        if not isinstance(entries, list):
            raise ValueError(f"Expected a list of view entries in {self.descriptions_path}, got {type(entries).__name__}")

        valid = []
        for entry in entries:
            label = entry.get("view_name") or entry.get("node_id")
            text = entry.get("short_description") or entry.get("description")
            if label and text:
                valid.append({"view_name": label, "short_description": text})
            else:
                print(
                    "[text_view_matcher] WARNING: skipping malformed entry (missing "
                    f"view_name/node_id or short_description/description): {entry}"
                )

        if not valid:
            raise FileNotFoundError(
                f"No usable view descriptions (with view_name/node_id + "
                f"short_description/description) found in {self.descriptions_path}."
            )
        return valid

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> np.ndarray:
        """Compute a single L2-normalized CLIP text embedding vector for one string."""
        self._ensure_model_loaded()
        import torch

        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)

        vector = features[0].cpu().numpy().astype(np.float32)
        return _normalize(vector)

    # ------------------------------------------------------------------
    # Index building / caching
    # ------------------------------------------------------------------

    def build_index(self, force: bool = False) -> None:
        """Populate self.index, either from the on-disk cache (if present and
        compatible) or by encoding every reference description from scratch.
        """
        if not force and self.cache_path and self.cache_path.exists():
            try:
                self.load_cache()
                print(
                    f"[text_view_matcher] Loaded {len(self.index)} cached reference "
                    f"text embeddings from {self.cache_path}"
                )
                return
            except Exception as e:
                print(f"[text_view_matcher] Cache unusable ({e}); rebuilding index.")

        entries = self._discover_reference_texts()
        self._ensure_model_loaded()

        self.index = []
        for entry in entries:
            view_name = entry["view_name"]
            text = entry["short_description"]
            embedding = self.encode_text(text)
            self.index.append(ReferenceTextEmbedding(view_name=view_name, text=text, embedding=embedding))
            print(f"[text_view_matcher] Encoded {view_name}")

        print(f"[text_view_matcher] Built index of {len(self.index)} view descriptions.")

        if self.cache_path:
            self.save_cache()

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "entries": [
                {"view_name": r.view_name, "text": r.text, "embedding": r.embedding}
                for r in self.index
            ],
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[text_view_matcher] Cached {len(self.index)} embeddings to {self.cache_path}")

    def load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            raise FileNotFoundError(f"No cache file at {self.cache_path}")
        with open(self.cache_path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("model_name") != self.model_name:
            raise ValueError(
                f"Cache was built with model {payload.get('model_name')!r}, but this "
                f"matcher is configured for {self.model_name!r}. Pass force=True to rebuild."
            )
        self.index = [
            ReferenceTextEmbedding(view_name=e["view_name"], text=e["text"], embedding=e["embedding"])
            for e in payload["entries"]
        ]

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, query_text: str, top_k: int = 5) -> dict:
        """Compare a query text against the reference text index.

        Returns a structured dict:
            {
              "query_text": str,
              "top_matches": [{"rank", "view_label", "reference_text", "similarity"}, ...],
              "best_view": str or None,
              "confidence_margin": float or None,
              "is_ambiguous": bool,
            }
        """
        if not self.index:
            self.build_index()

        query_embedding = self.encode_text(query_text)

        # Cosine similarity == dot product, since both sides are already L2-normalized.
        scored = [(ref, float(np.dot(query_embedding, ref.embedding))) for ref in self.index]
        scored.sort(key=lambda item: item[1], reverse=True)

        top_matches = [
            {
                "rank": i + 1,
                "view_label": ref.view_name,
                "reference_text": ref.text,
                "similarity": round(score, 6),
            }
            for i, (ref, score) in enumerate(scored[:top_k])
        ]

        best_view = top_matches[0]["view_label"] if top_matches else None
        confidence_margin = None
        is_ambiguous = True
        if len(top_matches) >= 2:
            confidence_margin = round(top_matches[0]["similarity"] - top_matches[1]["similarity"], 6)
            is_ambiguous = confidence_margin < AMBIGUITY_THRESHOLD
        elif len(top_matches) == 1:
            confidence_margin = top_matches[0]["similarity"]
            is_ambiguous = False

        return {
            "query_text": query_text,
            "top_matches": top_matches,
            "best_view": best_view,
            "confidence_margin": confidence_margin,
            "is_ambiguous": is_ambiguous,
        }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match a query text against a bank of reference-view text descriptions using CLIP text embeddings."
    )
    parser.add_argument("--query-text", required=True, help="Query text to match, e.g. an LLM's visual observation.")
    parser.add_argument("--descriptions", default=DEFAULT_DESCRIPTIONS_PATH, help="Path to view_descriptions.json.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top matches to show.")
    parser.add_argument("--cache", default=None, help="Path to a .pkl cache file for reference text embeddings.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Hugging Face CLIP model name.")
    parser.add_argument("--device", default=None, help="Force 'cpu' or 'cuda' (default: auto-detect).")
    parser.add_argument("--rebuild-cache", action="store_true", help="Recompute embeddings even if a cache file exists.")
    return parser


def main(argv=None) -> dict:
    args = _build_arg_parser().parse_args(argv)

    matcher = TextViewMatcher(
        descriptions_path=args.descriptions,
        model_name=args.model,
        cache_path=args.cache,
        device=args.device,
    )
    matcher.build_index(force=args.rebuild_cache)
    result = matcher.match(args.query_text, top_k=args.top_k)

    print()
    ambiguous_note = "  (AMBIGUOUS)" if result["is_ambiguous"] else ""
    print(f"Best view: {result['best_view']}")
    print(f"Confidence margin: {result['confidence_margin']}{ambiguous_note}")

    print(f"\nTop {len(result['top_matches'])} matches:")
    for m in result["top_matches"]:
        print(f"  {m['rank']}. {m['view_label']:<20s} {m['similarity']:.4f}  {m['reference_text'][:80]}")

    return result


if __name__ == "__main__":
    main()
