"""
Image-embedding based reference-view matcher.

Compares a query screenshot (e.g. the VTK render's current camera view) against a
bank of reference-view images using CLIP image embeddings + cosine similarity, and
returns the most visually similar reference view(s). This is a *visual* alternative
to the existing text-based diagnosis in session.py/prompt_writer.py — it does not
touch that pipeline at all (self-contained, no relative imports from the rest of
camera_reasoning, so it also runs fine as a standalone script).

Reference image layout (both are supported, and can be mixed):
  reference_views/
    Dorsal.png                 <- flat file: view_label = "Dorsal"
    Plantar.png
    dorsal/
      dorsal_0.png              <- subfolder: view_label = "dorsal", multiple refs
      dorsal_1.png

CLI usage:
  python camera_reasoning/image_view_matcher.py \\
      --query generated_code/current_screenshot.png \\
      --references reference_views \\
      --top-k 5 \\
      --cache reference_embeddings.pkl \\
      --contact-sheet match_results.png

This module also provides GraphImageMatcher/GraphReferenceEmbedding, the
node_id-based counterpart used by the camera-relative spatial graph
(camera_spatial_graph.py) — see the section below. Unlike ImageViewMatcher
(which infers a semantic view_label from filenames/subfolders), GraphImageMatcher
identifies references purely by neutral node_id, loaded directly from a
CameraSpatialGraph (itself built from camera_graph.json / camera_nodes.json) — it
never infers an ID from a filename. ImageViewMatcher/ReferenceEmbedding are left
unchanged so the existing semantic-label pipeline keeps working as-is.
"""
import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

# confidence_margin below this is flagged ambiguous (best/second-best views are too
# close to be confident). Tune to taste once you see real margins on your data.
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
class ReferenceEmbedding:
    image_path: str
    view_label: str
    embedding: np.ndarray  # L2-normalized, shape (D,)


class ImageViewMatcher:
    """Matches a query image against a bank of reference-view images via CLIP
    image embeddings and cosine similarity.

    Typical usage:
        matcher = ImageViewMatcher("reference_views", cache_path="reference_embeddings.pkl")
        matcher.build_index()
        result = matcher.match("current_screenshot.png", top_k=5)
    """

    def __init__(
        self,
        reference_dir: str,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.reference_dir = Path(reference_dir)
        if not self.reference_dir.exists():
            raise FileNotFoundError(
                f"Reference image directory not found: {self.reference_dir}. "
                "Generate reference views first (see camera_reasoning/reference_views.py), "
                "or pass the correct --references path."
            )

        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = _resolve_device(device)

        self._model = None
        self._processor = None
        self.index: List[ReferenceEmbedding] = []

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

        print(f"[image_view_matcher] Loading {self.model_name!r} on {self.device}...")
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    # ------------------------------------------------------------------
    # Reference discovery
    # ------------------------------------------------------------------

    def _discover_reference_images(self) -> List[Tuple[str, Path]]:
        """Return [(view_label, image_path), ...] for every reference image found.

        Supports both flat files ("dorsal.png" -> label "dorsal") and per-view
        subfolders ("dorsal/dorsal_0.png" -> label "dorsal", one entry per image).
        Anything that isn't a recognized image file (metadata JSON, .DS_Store, ...)
        is silently skipped.
        """
        pairs: List[Tuple[str, Path]] = []
        for entry in sorted(self.reference_dir.iterdir()):
            if entry.is_dir():
                view_label = entry.name
                for image_path in sorted(entry.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        pairs.append((view_label, image_path))
            elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
                pairs.append((entry.stem, entry))

        if not pairs:
            raise FileNotFoundError(
                f"No reference images (.png/.jpg/.jpeg) found under {self.reference_dir}. "
                "Expected flat files like 'dorsal.png' and/or subfolders like "
                "'dorsal/dorsal_0.png'."
            )
        return pairs

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_image(self, image_path: str) -> np.ndarray:
        """Compute a single L2-normalized CLIP embedding vector for one image."""
        self._ensure_model_loaded()
        import torch

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        vector = features[0].cpu().numpy().astype(np.float32)
        return _normalize(vector)

    # ------------------------------------------------------------------
    # Index building / caching
    # ------------------------------------------------------------------

    def build_index(self, force: bool = False) -> None:
        """Populate self.index, either from the on-disk cache (if present and
        compatible) or by encoding every reference image from scratch.
        """
        if not force and self.cache_path and self.cache_path.exists():
            try:
                self.load_cache()
                print(
                    f"[image_view_matcher] Loaded {len(self.index)} cached reference "
                    f"embeddings from {self.cache_path}"
                )
                return
            except Exception as e:
                print(f"[image_view_matcher] Cache unusable ({e}); rebuilding index.")

        pairs = self._discover_reference_images()
        self._ensure_model_loaded()

        self.index = []
        for view_label, image_path in pairs:
            embedding = self.encode_image(str(image_path))
            self.index.append(
                ReferenceEmbedding(image_path=str(image_path), view_label=view_label, embedding=embedding)
            )
            print(f"[image_view_matcher] Encoded {view_label}: {image_path.name}")

        num_views = len(set(r.view_label for r in self.index))
        print(f"[image_view_matcher] Built index of {len(self.index)} images across {num_views} views.")

        if self.cache_path:
            self.save_cache()

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "entries": [
                {"image_path": r.image_path, "view_label": r.view_label, "embedding": r.embedding}
                for r in self.index
            ],
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[image_view_matcher] Cached {len(self.index)} embeddings to {self.cache_path}")

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
            ReferenceEmbedding(image_path=e["image_path"], view_label=e["view_label"], embedding=e["embedding"])
            for e in payload["entries"]
        ]

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, query_image_path: str, top_k: int = 5) -> dict:
        """Compare a query screenshot against the reference index.

        Returns a structured dict:
            {
              "query_image": str,
              "top_matches": [{"rank", "view_label", "reference_image", "similarity"}, ...],
              "view_scores": [{"view_label", "max_similarity", "mean_similarity", "num_references"}, ...],
              "best_view": str or None,
              "confidence_margin": float or None,
              "is_ambiguous": bool,
            }
        """
        if not self.index:
            self.build_index()

        query_embedding = self.encode_image(query_image_path)

        # Cosine similarity == dot product, since both sides are already L2-normalized.
        scored = [(ref, float(np.dot(query_embedding, ref.embedding))) for ref in self.index]
        scored.sort(key=lambda item: item[1], reverse=True)

        top_matches = [
            {
                "rank": i + 1,
                "view_label": ref.view_label,
                "reference_image": ref.image_path,
                "similarity": round(score, 6),
            }
            for i, (ref, score) in enumerate(scored[:top_k])
        ]

        view_scores = _aggregate_by_view(scored)

        best_view = view_scores[0]["view_label"] if view_scores else None
        confidence_margin = None
        is_ambiguous = True
        if len(view_scores) >= 2:
            confidence_margin = round(view_scores[0]["max_similarity"] - view_scores[1]["max_similarity"], 6)
            is_ambiguous = confidence_margin < AMBIGUITY_THRESHOLD
        elif len(view_scores) == 1:
            confidence_margin = view_scores[0]["max_similarity"]
            is_ambiguous = False

        return {
            "query_image": str(query_image_path),
            "top_matches": top_matches,
            "view_scores": view_scores,
            "best_view": best_view,
            "confidence_margin": confidence_margin,
            "is_ambiguous": is_ambiguous,
        }


def _aggregate_by_view(scored: List[Tuple[ReferenceEmbedding, float]]) -> List[dict]:
    """Group per-image similarity scores by view_label and compute max/mean per view.
    Returns view_scores sorted by max_similarity, descending.
    """
    by_view: Dict[str, List[float]] = {}
    for ref, score in scored:
        by_view.setdefault(ref.view_label, []).append(score)

    view_scores = [
        {
            "view_label": view_label,
            "max_similarity": round(max(scores), 6),
            "mean_similarity": round(float(np.mean(scores)), 6),
            "num_references": len(scores),
        }
        for view_label, scores in by_view.items()
    ]
    view_scores.sort(key=lambda v: v["max_similarity"], reverse=True)
    return view_scores


def save_contact_sheet(query_image_path: str, top_matches: List[dict], output_path: str, thumb_size: int = 256) -> None:
    """Save a horizontal strip: the query image, followed by each top-k reference
    match, each labeled with its rank/label/similarity — useful for eyeballing
    whether the matcher picked a sensible view.

    Works with both ImageViewMatcher.match() top_matches (keyed by "view_label")
    and GraphImageMatcher.match() top_matches (keyed by "node_id").
    """
    from PIL import ImageDraw

    query_img = Image.open(query_image_path).convert("RGB").resize((thumb_size, thumb_size))
    tiles = [("QUERY", query_img)]
    for m in top_matches:
        img = Image.open(m["reference_image"]).convert("RGB").resize((thumb_size, thumb_size))
        match_label = m.get("view_label", m.get("node_id"))
        label = f"{m['rank']}. {match_label} ({m['similarity']:.3f})"
        tiles.append((label, img))

    label_height = 22
    sheet = Image.new("RGB", (thumb_size * len(tiles), thumb_size + label_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(tiles):
        x = i * thumb_size
        sheet.paste(img, (x, label_height))
        draw.text((x + 4, 4), label, fill=(255, 255, 255))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


# ------------------------------------------------------------------
# Graph-native (node_id-based) matcher — camera-relative spatial graph
# ------------------------------------------------------------------

@dataclass
class GraphReferenceEmbedding:
    node_id: str
    image_path: str
    embedding: np.ndarray  # L2-normalized, shape (D,)


def _graph_fingerprint(nodes: Dict[str, dict]) -> str:
    """Stable hash over (node_id, image_path, image_mtime) triples. Used to
    invalidate a GraphImageMatcher's embedding cache automatically if the
    underlying graph, its image set, OR an image's on-disk content changes —
    without requiring the caller to remember to pass force=True. The mtime is
    what catches the case where camera_relative_views.py (or any other
    generator) rewrites the same file paths with different content (e.g. a
    different dataset), which node_id/image_path alone would miss.
    """
    entries = []
    for node_id, data in nodes.items():
        image_path = data.get("image_path")
        try:
            mtime = Path(image_path).stat().st_mtime
        except (OSError, TypeError):
            mtime = None
        entries.append((node_id, image_path, mtime))
    payload = json.dumps(sorted(entries))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GraphImageMatcher:
    """Image-embedding matcher keyed by neutral graph node_id, the node_id-based
    counterpart to ImageViewMatcher for the camera-relative spatial graph
    (camera_spatial_graph.CameraSpatialGraph). Does NOT involve an LLM anywhere
    in matching: query image -> image embedding -> nearest graph node is a
    fully deterministic CLIP-embedding + cosine-similarity lookup.

    Typical usage:
        graph = CameraSpatialGraph.from_json("reference_views_relative/camera_graph.json")
        matcher = GraphImageMatcher(graph, cache_path="reference_views_relative/graph_embeddings.pkl")
        matcher.build_index()
        result = matcher.match("current_screenshot.png", top_k=5)
        route = graph.route(result["best_node_id"], target_node_id)
    """

    def __init__(
        self,
        graph,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.graph = graph
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = _resolve_device(device)

        self._model = None
        self._processor = None
        self.index: List[GraphReferenceEmbedding] = []
        self._fingerprint = _graph_fingerprint(graph.nodes)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        print(f"[image_view_matcher] Loading {self.model_name!r} on {self.device}...")
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    def encode_image(self, image_path: str) -> np.ndarray:
        self._ensure_model_loaded()
        import torch

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        vector = features[0].cpu().numpy().astype(np.float32)
        return _normalize(vector)

    def build_index(self, force: bool = False) -> None:
        if not force and self.cache_path and self.cache_path.exists():
            try:
                self.load_cache()
                print(f"[image_view_matcher] Loaded {len(self.index)} cached graph-node embeddings from {self.cache_path}")
                return
            except Exception as e:
                print(f"[image_view_matcher] Cache unusable ({e}); rebuilding graph index.")

        self._ensure_model_loaded()
        self.index = []
        for node_id, data in self.graph.nodes.items():
            image_path = data["image_path"]
            embedding = self.encode_image(image_path)
            self.index.append(GraphReferenceEmbedding(node_id=node_id, image_path=image_path, embedding=embedding))
            print(f"[image_view_matcher] Encoded {node_id}: {Path(image_path).name}")

        print(f"[image_view_matcher] Built graph index of {len(self.index)} nodes.")
        if self.cache_path:
            self.save_cache()

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "graph_version": getattr(self.graph, "version", None),
            "graph_fingerprint": self._fingerprint,
            "entries": [
                {"node_id": r.node_id, "image_path": r.image_path, "embedding": r.embedding}
                for r in self.index
            ],
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[image_view_matcher] Cached {len(self.index)} graph-node embeddings to {self.cache_path}")

    def load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            raise FileNotFoundError(f"No cache file at {self.cache_path}")
        with open(self.cache_path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("model_name") != self.model_name:
            raise ValueError(
                f"Cache was built with model {payload.get('model_name')!r}, but this "
                f"matcher is configured for {self.model_name!r}."
            )
        if payload.get("graph_fingerprint") != self._fingerprint:
            raise ValueError(
                "Cache was built from a different graph/image set (node IDs or image "
                "paths changed) — stale, must be rebuilt."
            )
        self.index = [
            GraphReferenceEmbedding(node_id=e["node_id"], image_path=e["image_path"], embedding=e["embedding"])
            for e in payload["entries"]
        ]

    def match(self, query_image_path: str, top_k: int = 5) -> dict:
        """Compare a query screenshot against the graph-node index.

        Returns:
            {
              "query_image": str,
              "best_node_id": str or None,
              "confidence_margin": float or None,
              "is_ambiguous": bool,
              "top_matches": [{"rank", "node_id", "reference_image", "similarity"}, ...],
            }
        """
        if not self.index:
            self.build_index()

        query_embedding = self.encode_image(query_image_path)
        scored = [(ref, float(np.dot(query_embedding, ref.embedding))) for ref in self.index]
        scored.sort(key=lambda item: item[1], reverse=True)

        top_matches = [
            {
                "rank": i + 1,
                "node_id": ref.node_id,
                "reference_image": ref.image_path,
                "similarity": round(score, 6),
            }
            for i, (ref, score) in enumerate(scored[:top_k])
        ]

        best_node_id = top_matches[0]["node_id"] if top_matches else None
        confidence_margin = None
        is_ambiguous = True
        if len(top_matches) >= 2:
            confidence_margin = round(top_matches[0]["similarity"] - top_matches[1]["similarity"], 6)
            is_ambiguous = confidence_margin < AMBIGUITY_THRESHOLD
        elif len(top_matches) == 1:
            confidence_margin = top_matches[0]["similarity"]
            is_ambiguous = False

        return {
            "query_image": str(query_image_path),
            "best_node_id": best_node_id,
            "confidence_margin": confidence_margin,
            "is_ambiguous": is_ambiguous,
            "top_matches": top_matches,
        }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match a query screenshot against a bank of reference-view images using CLIP embeddings."
    )
    parser.add_argument("--query", required=True, help="Path to the query screenshot image.")
    parser.add_argument("--references", required=True, help="Path to the reference images directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top individual matches to show.")
    parser.add_argument("--cache", default=None, help="Path to a .pkl cache file for reference embeddings.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Hugging Face CLIP model name.")
    parser.add_argument("--device", default=None, help="Force 'cpu' or 'cuda' (default: auto-detect).")
    parser.add_argument("--rebuild-cache", action="store_true", help="Recompute embeddings even if a cache file exists.")
    parser.add_argument(
        "--contact-sheet", default=None,
        help="Optional path to save a query-vs-top-k visualization PNG (e.g. match_results.png).",
    )
    return parser


def main(argv=None) -> dict:
    args = _build_arg_parser().parse_args(argv)

    matcher = ImageViewMatcher(
        reference_dir=args.references,
        model_name=args.model,
        cache_path=args.cache,
        device=args.device,
    )
    matcher.build_index(force=args.rebuild_cache)
    result = matcher.match(args.query, top_k=args.top_k)

    print()
    ambiguous_note = "  (AMBIGUOUS)" if result["is_ambiguous"] else ""
    print(f"Best view: {result['best_view']}")
    print(f"Confidence margin: {result['confidence_margin']}{ambiguous_note}")

    print(f"\nTop {len(result['top_matches'])} individual matches:")
    for m in result["top_matches"]:
        print(f"  {m['rank']}. {m['view_label']:<20s} {m['similarity']:.4f}  {m['reference_image']}")

    print("\nAggregated view scores:")
    for v in result["view_scores"]:
        print(
            f"  {v['view_label']:<20s} max={v['max_similarity']:.4f}  "
            f"mean={v['mean_similarity']:.4f}  n={v['num_references']}"
        )

    if args.contact_sheet:
        save_contact_sheet(args.query, result["top_matches"], args.contact_sheet)
        print(f"\nSaved contact sheet to {args.contact_sheet}")

    return result


if __name__ == "__main__":
    main()
