"""Optional offline FastEmbed retrieval, fused with the existing lexical index."""

import hashlib
import json
from pathlib import Path


class SemanticSearch:
    def __init__(self, model_dir: Path, data_dir: Path):
        # Explicit local_files_only prevents confidential execution from causing
        # an implicit model download or network lookup.
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ValueError(
                "Install the optional semantic dependency with uv sync --extra semantic."
            ) from None

        digest = hashlib.sha256()
        for path in sorted(model_dir.rglob("*")):
            if path.is_file() and path.suffix in {".onnx", ".json", ".txt"}:
                digest.update(str(path.relative_to(model_dir)).encode())
                with path.open("rb") as handle:
                    digest.update(hashlib.file_digest(handle, "sha256").digest())
        self.model_sha256 = digest.hexdigest()

        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=str(model_dir),
            threads=4,
            local_files_only=True,
        )
        self.directory = data_dir / "embeddings"
        self.directory.mkdir(exist_ok=True)
        self.vectors = None
        self.key = None

    def prepare(self, index):
        import numpy as np

        identity = (
            index.fingerprint
            + self.model_sha256
            + "\n"
            + "\n".join(p.id + p.text for p in index.passages)
        )
        key = hashlib.sha256(identity.encode()).hexdigest()
        if key == self.key:
            return
        path = self.directory / (key + ".npy")
        if path.exists():
            vectors = np.load(path, allow_pickle=False)
        else:
            vectors = np.array(
                list(
                    self.model.passage_embed(
                        [p.heading + "\n" + p.text for p in index.passages], batch_size=16
                    )
                )
            )
            # Exclusive creation preserves previous cached artifacts.
            with path.open("xb") as handle:
                np.save(handle, vectors, allow_pickle=False)
        if vectors.shape != (len(index.passages), 384):
            raise ValueError("Invalid embedding cache shape.")
        self.vectors, self.key = vectors, key
        manifest = self.directory / (key + ".json")
        if not manifest.exists():
            manifest.write_text(
                json.dumps(
                    {
                        "document_sha256": index.fingerprint,
                        "model": "BAAI/bge-small-en-v1.5",
                        "dimensions": 384,
                        "passages": len(index.passages),
                        "offline": True,
                        "model_files_sha256": self.model_sha256,
                    }
                ),
                encoding="utf-8",
            )

    def search(self, query, index, limit):
        import numpy as np

        if self.key is None:
            raise ValueError("Semantic index has not been prepared.")
        vector = np.array(next(self.model.query_embed(query)))
        scores = self.vectors @ vector
        dense = [index.passages[int(n)] for n in np.argsort(-scores)[:20]]
        lexical = index.search(query, limit=20)
        fused = {}
        for branch in (lexical, dense):
            for rank, passage in enumerate(branch, 1):
                fused[passage.id] = fused.get(passage.id, 0) + 1 / (60 + rank)
        sources = {p.id: p for p in index.passages}
        return [
            sources[key] for key in sorted(fused, key=fused.get, reverse=True)[:limit]
        ]
