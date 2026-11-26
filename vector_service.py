"""PA small vector search service using FAISS + SQLite.

Run:
  python vector_service.py                     # FAISS if installed, NumPy fallback
  python vector_service.py --backend faiss     # require faiss-cpu
  python vector_service.py --benchmark --plots # compare exact and ANN indexes

Requires numpy. FAISS requires pip install faiss-cpu; plots need matplotlib.
SQLite is part of Python. The content encoder comes from article1_numpy_lab.py.
This is an educational demo, not a production database or authorization system.
"""
from __future__ import annotations
import argparse
import csv
import sqlite3
import time
from pathlib import Path
import numpy as np
from vector_search import SENTENCES, QUERY, CONCEPTS, embed, unit

try:
    import faiss
except ImportError:
    faiss = None

D = len(CONCEPTS)
MODEL_VERSION = "toy-concepts-v1"
ALPHA_IDS = {1, 2, 3, 4, 5, 12, 14}


def normalized_32(text: str) -> np.ndarray:
    return np.ascontiguousarray(unit(embed(text)).astype(np.float32))


def open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        doc_id TEXT NOT NULL UNIQUE,
        content TEXT NOT NULL,
        owner TEXT NOT NULL,
        category TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        embedding BLOB NOT NULL,
        model_version TEXT NOT NULL
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_owner_active ON documents(owner, active)")
    return conn


def reset_demo_store(conn: sqlite3.Connection) -> None:
    """Repeatable seed. This clears ONLY the dedicated demo documents table."""
    with conn:
        conn.execute("DELETE FROM documents")
        for i, sentence in enumerate(SENTENCES, 1):
            owner = "alpha" if i in ALPHA_IDS else "beta"
            category = "energy" if i in {1,2,3,4,5,7,12,13,14} else "operations"
            v = normalized_32(sentence)
            conn.execute("""INSERT INTO documents
                (id, doc_id, content, owner, category, active, embedding, model_version)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
                (i, f"S{i:02d}", sentence, owner, category,
                 sqlite3.Binary(v.tobytes()), MODEL_VERSION))


def active_rows(conn: sqlite3.Connection, owner: str | None = None):
    query = "SELECT * FROM documents WHERE active = 1"
    params = ()
    if owner is not None:
        query += " AND owner = ?"
        params = (owner,)
    query += " ORDER BY id"
    rows = conn.execute(query, params).fetchall()
    if any(r["model_version"] != MODEL_VERSION for r in rows):
        raise RuntimeError("Index and stored embeddings use different versions")
    return rows


def matrix_from_rows(rows):
    if not rows:
        return np.empty((0, D), dtype=np.float32)
    mat = np.vstack([
        np.frombuffer(r["embedding"], dtype=np.float32).copy() for r in rows
    ])
    if mat.shape[1] != D:
        raise RuntimeError("Embedding dimension mismatch")
    return np.ascontiguousarray(mat, dtype=np.float32)


class NumpyFlat:
    """Exact-search fallback used to validate storage without a FAISS install."""
    def __init__(self, X, ids):
        self.X = X
        self.ids = ids
        self.ntotal = len(ids)

    def search(self, questions: np.ndarray, k: int):
        similarities = questions @ self.X.T
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(similarities, order, axis=1), self.ids[order]


def build_index(conn, backend: str):
    rows = active_rows(conn)
    vectors = matrix_from_rows(rows)
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    if backend == "faiss":
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(D))
        if len(vectors):
            index.add_with_ids(vectors, ids)
    else:
        index = NumpyFlat(vectors, ids)
    return index


def search_global(index, conn, question: str, k: int = 5):
    q = normalized_32(question)
    if np.linalg.norm(q) == 0:
        return []
    if index.ntotal == 0:
        return []
    distances, ids = index.search(q.reshape(1, -1), min(k, index.ntotal))
    answers = []
    for score, doc_id in zip(distances[0], ids[0]):
        if doc_id < 0:
            continue
        row = conn.execute("SELECT * FROM documents WHERE id=? AND active=1", (int(doc_id),)).fetchone()
        if row is not None:
            answers.append({"doc_id": row["doc_id"], "owner": row["owner"],
                            "score": float(score), "content": row["content"]})
    return answers


def search_authorized_exact(conn, question: str, owner: str, k: int = 3):
    """Restrict candidates BEFORE scoring. Works for this tiny repository.

    In production, implement an access-control policy in the retrieval service
    and use a store with a supported filter-aware search index.
    """
    rows = active_rows(conn, owner=owner)
    if not rows:
        return []
    X = matrix_from_rows(rows)
    q = normalized_32(question)
    if np.linalg.norm(q) == 0:
        return []
    scores = X @ q
    order = np.argsort(-scores, kind="stable")[:k]
    return [{"doc_id": rows[i]["doc_id"], "owner": rows[i]["owner"],
             "score": float(scores[i]), "content": rows[i]["content"]}
            for i in order]


def update_document(conn, doc_id: str, new_content: str) -> None:
    """DB is source of truth; caller rebuilds index as part of demo workflow."""
    vector = normalized_32(new_content)
    with conn:
        cur = conn.execute("""UPDATE documents
            SET content=?, embedding=?, model_version=?
            WHERE doc_id=? AND active=1""",
            (new_content, sqlite3.Binary(vector.tobytes()), MODEL_VERSION, doc_id))
        if cur.rowcount != 1:
            raise KeyError(doc_id)


def deactivate_document(conn, doc_id: str) -> None:
    with conn:
        conn.execute("UPDATE documents SET active=0 WHERE doc_id=?", (doc_id,))


def print_hits(label, hits):
    print(f"\n{label}")
    for item in hits:
        print(f" {item['doc_id']:3s} {item['score']:.4f}  {item['content']}")


def demo(path: Path, backend: str):
    conn = open_store(path / "lab.sqlite3")
    reset_demo_store(conn)
    index = build_index(conn, backend)
    print(f"Backend: {backend}, vectors: {index.ntotal}, dimensions: {D}")
    top3 = search_global(index, conn, QUERY, k=3)
    print_hits("Global Top 3", top3)
    # UNSAFE retrieval pattern: filter only AFTER globally retrieving Top K.
    naive = [r for r in top3 if r["owner"] == "alpha"]
    print_hits("Naive post-filtered Top 3 (incomplete)", naive)
    # Correct result set for small demo: select authorized candidates first.
    restricted = search_authorized_exact(conn, QUERY, owner="alpha", k=3)
    print_hits("Owner alpha: filter BEFORE exact search", restricted)
    if len(naive) >= len(restricted):
        print("Note: this example expected naive post-filtering to miss one record.")

    if backend == "faiss":
        index_path = path / "energy_flat.faiss"
        faiss.write_index(index, str(index_path))
        # Only read indexes from trusted sources. FAISS does not validate integrity.
        restored = faiss.read_index(str(index_path))
        assert restored.ntotal == index.ntotal
        print(f"\nSaved and reloaded FAISS index: {index_path}")

    update_document(conn, "S02", "Solar batteries store electricity to power lamps after dark.")
    index = build_index(conn, backend)
    print_hits("After modifying S02 and rebuilding", search_global(index, conn, QUERY, 3))
    deactivate_document(conn, "S13")
    index = build_index(conn, backend)
    hits = search_global(index, conn, QUERY, 5)
    assert all(r["doc_id"] != "S13" for r in hits)
    print_hits("After deactivating S13 and rebuilding", hits)
    if backend == "faiss":
        faiss.write_index(index, str(path / "energy_flat.faiss"))
    conn.close()


def benchmark(path: Path, plots: bool, n: int = 10000, d: int = 64):
    if faiss is None:
        print("FAISS is unavailable; skip ANN benchmark. Install: pip install faiss-cpu")
        return
    # Synthetic numerical data ONLY: does not benchmark real semantic quality.
    faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(2026)
    X = np.ascontiguousarray(rng.normal(size=(n, d)).astype(np.float32))
    faiss.normalize_L2(X)
    probes = np.ascontiguousarray(X[rng.choice(n, size=100, replace=False)] +
                                  .09 * rng.normal(size=(100, d)).astype(np.float32))
    faiss.normalize_L2(probes)
    k = 10
    flat = faiss.IndexFlatIP(d)
    flat.add(X)
    # Exact reference neighbours for measuring ANN recall, not semantic precision.
    exact_scores, exact_ids = flat.search(probes, k)
    hnsw = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    hnsw.add(X)
    clusters = min(64, max(4, n // 60))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, clusters,
                             faiss.METRIC_INNER_PRODUCT)
    ivf.train(X[:min(n, max(3000, 50 * clusters))])
    ivf.add(X)

    def measure(name, idx):
        idx.search(probes, k)  # warm up
        t0 = time.perf_counter()
        _, found = idx.search(probes, k)
        total_ms = (time.perf_counter() - t0) * 1000
        recall = sum(len(set(found[i]) & set(exact_ids[i]))
                     for i in range(len(probes))) / (len(probes) * k)
        row = {"index": name, "vectors": n, "dimension": d,
               "recall_at_10": recall, "batch_mean_ms_per_query": total_ms / len(probes)}
        print(f"{name:16s} recall@10={recall:.3f} batch mean ms/query={total_ms / len(probes):.5f}")
        return row

    results = [measure("Flat exact", flat)]
    for ef in [8, 16, 32, 64]:
        hnsw.hnsw.efSearch = ef
        results.append(measure(f"HNSW ef={ef}", hnsw))
    for nprobe in [1, 4, 16, 32]:
        ivf.nprobe = min(nprobe, clusters)
        results.append(measure(f"IVF nprobe={nprobe}", ivf))

    with (path / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)

    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        for family in ("Flat", "HNSW", "IVF"):
            rows = [x for x in results if x["index"].startswith(family)]
            plt.plot([x["batch_mean_ms_per_query"] for x in rows],
                     [x["recall_at_10"] for x in rows], marker="o", label=family)
            for x in rows:
                plt.annotate(x["index"],
                             (x["batch_mean_ms_per_query"], x["recall_at_10"]),
                             fontsize=7, xytext=(4, 4), textcoords="offset points")
        plt.xlabel("Mean time per query in batch (ms)")
        plt.ylabel("Recall@10 relative to exact FAISS Flat")
        plt.ylim(-.03, 1.05); plt.legend(); plt.grid(alpha=.2)
        plt.title(f"Synthetic index comparison, n={n}, d={d}; not a production benchmark")
        plt.tight_layout(); plt.savefig(path / "08_faiss_recall_latency.png", dpi=150)
        plt.close()
    print("\nThese timings are single-thread batched experiments, not online p95 latency.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("vector_medium_output"))
    parser.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    backend = ("faiss" if faiss is not None else "numpy") if args.backend == "auto" else args.backend
    if backend == "faiss" and faiss is None:
        parser.error("FAISS is missing. Run: python -m pip install faiss-cpu")
    if backend == "numpy" and faiss is None:
        print("FAISS is not installed. Running exact NumPy fallback for the database operations.")
    demo(args.out, backend)
    if args.benchmark:
        benchmark(args.out, args.plots)
    print("\nOutputs saved at:", args.out.resolve())


if __name__ == "__main__":
    main()
