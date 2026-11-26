"""Vector database operations with FAISS + SQLite 

Examples:
    python vector_service.py --plots
    python vector_service.py --backend numpy --plots
    python vector_service.py --backend faiss --plots
    python vector_service.py --backend faiss --benchmark --plots
    python vector_service.py --backend faiss --benchmark --plots --bench-n 5000

Dependencies: numpy; matplotlib for --plots; optional faiss-cpu for FAISS/ANN.
SQLite is included with Python. No ML framework or online service is needed.

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
MODEL_VERSION = "vector-concepts-v1"
ALPHA_IDS = {1, 2, 3, 4, 5, 12, 14}


def normalized_32(text: str) -> np.ndarray:
    # Encode and L2-normalize 
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_docs_owner_active ON documents(owner, active)"
    )
    return conn


def reset_demo_store(conn: sqlite3.Connection) -> None:
    # Repeatable seed - clears this `documents` table.
    with conn:
        conn.execute("DELETE FROM documents")
        for i, sentence in enumerate(SENTENCES, 1):
            owner = "ABC" if i in ALPHA_IDS else "XYZ"
            category = "energy" if i in {1, 2, 3, 4, 5, 7, 12, 13, 14} else "operations"
            vector = normalized_32(sentence)
            conn.execute("""INSERT INTO documents
                (id, doc_id, content, owner, category, active, embedding, model_version)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
                (i, f"S{i:02d}", sentence, owner, category,
                 sqlite3.Binary(vector.tobytes()), MODEL_VERSION))


def active_rows(conn: sqlite3.Connection, owner: str | None = None):
    query = "SELECT * FROM documents WHERE active = 1"
    params: tuple[str, ...] = ()
    if owner is not None:
        query += " AND owner = ?"
        params = (owner,)
    rows = conn.execute(query + " ORDER BY id", params).fetchall()
    if any(row["model_version"] != MODEL_VERSION for row in rows):
        raise RuntimeError("Stored embeddings and index encoder versions differ")
    return rows


def matrix_from_rows(rows) -> np.ndarray:
    if not rows:
        return np.empty((0, D), dtype=np.float32)
    matrix = np.vstack([
        np.frombuffer(row["embedding"], dtype=np.float32).copy() for row in rows
    ])
    if matrix.shape[1] != D:
        raise RuntimeError("Embedding dimension mismatch")
    return np.ascontiguousarray(matrix, dtype=np.float32)


class NumpyFlat:
    # Exact cosine (normalized-inner-product) fallback.

    def __init__(self, vectors: np.ndarray, ids: np.ndarray):
        self.vectors = vectors
        self.ids = ids
        self.ntotal = len(ids)

    def search(self, queries: np.ndarray, k: int):
        similarities = queries @ self.vectors.T
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        return (np.take_along_axis(similarities, order, axis=1), self.ids[order])


def build_index(conn: sqlite3.Connection, backend: str):
    rows = active_rows(conn)
    vectors = matrix_from_rows(rows)
    ids = np.array([row["id"] for row in rows], dtype=np.int64)
    if backend == "faiss":
        if faiss is None:
            raise RuntimeError("Install FAISS: python -m pip install faiss-cpu")
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(D))
        if len(vectors):
            index.add_with_ids(vectors, ids)
    else:
        index = NumpyFlat(vectors, ids)
    return index


def search_global(index, conn: sqlite3.Connection, question: str, k: int = 5):
    # Used for searching in Vector
    query = normalized_32(question)
    if np.linalg.norm(query) == 0 or index.ntotal == 0:
        return []
    scores, ids = index.search(query.reshape(1, -1), min(k, index.ntotal))
    result = []
    for score, numeric_id in zip(scores[0], ids[0]):
        if numeric_id < 0:
            continue
        row = conn.execute(
            "SELECT * FROM documents WHERE id=? AND active=1", (int(numeric_id),)
        ).fetchone()
        if row is not None:
            result.append({"doc_id": row["doc_id"], "owner": row["owner"],
                           "score": float(score), "content": row["content"]})
    return result


def search_authorized_exact(conn: sqlite3.Connection, question: str,
                            owner: str, k: int = 3):
    ## Filter BEFORE searching; a method for small datasets.

    
    rows = active_rows(conn, owner=owner)
    if not rows:
        return []
    query = normalized_32(question)
    if np.linalg.norm(query) == 0:
        return []
    scores = matrix_from_rows(rows) @ query
    order = np.argsort(-scores, kind="stable")[:k]
    return [{"doc_id": rows[i]["doc_id"], "owner": rows[i]["owner"],
             "score": float(scores[i]), "content": rows[i]["content"]}
            for i in order]


def score_document(conn: sqlite3.Connection, doc_id: str, question: str) -> float:
    # Score stored vector directly, including before/after an update.
    row = conn.execute(
        "SELECT embedding FROM documents WHERE doc_id=? AND active=1", (doc_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Active record not found: {doc_id}")
    vector = np.frombuffer(row["embedding"], dtype=np.float32)
    return float(vector @ normalized_32(question))


def update_document(conn: sqlite3.Connection, doc_id: str, new_content: str) -> None:
    # SQLite is source of truth - rebuild index after updating the demo.
    vector = normalized_32(new_content)
    with conn:
        cur = conn.execute("""UPDATE documents
            SET content=?, embedding=?, model_version=?
            WHERE doc_id=? AND active=1""",
            (new_content, sqlite3.Binary(vector.tobytes()), MODEL_VERSION, doc_id))
        if cur.rowcount != 1:
            raise KeyError(doc_id)


def deactivate_document(conn: sqlite3.Connection, doc_id: str) -> None:
    with conn:
        cur = conn.execute("UPDATE documents SET active=0 WHERE doc_id=? AND active=1",
                           (doc_id,))
        if cur.rowcount != 1:
            raise KeyError(doc_id)


def print_hits(title: str, hits: list[dict]) -> None:
    print("\n" + title)
    for hit in hits:
        print(f" {hit['doc_id']:3s} {hit['score']:.4f}  {hit['content']}")


def write_csv(path: Path, headers: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def plot_demo_results(path: Path, global_top3: list[dict],
                      postfiltered: list[dict], authorized_top3: list[dict],
                      before_s02: float, after_s02: float,
                      starting_count: int, final_count: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(filename: str):
        plt.tight_layout(pad=0.7)
        plt.savefig(path / filename, dpi=180, bbox_inches="tight", pad_inches=0.06)
        plt.close()
        print(f"Saved plot: {filename}")

    # Figure 5 — global top-3 -> naive post-filter -> correct filter-first top-3.
    groups = [
        ("Global top-3", global_top3),
        ("Post-filtered to ABC", postfiltered),
        ("Filter before search", authorized_top3),
    ]
    y_values = []
    row_labels = []
    score_values = []
    owner_labels = []
    separators = []
    ypos = 0
    for group_name, hits in groups:
        for hit in hits:
            y_values.append(ypos)
            row_labels.append(f"{group_name}: {hit['doc_id']} ({hit['owner']})")
            score_values.append(hit["score"])
            owner_labels.append(hit["owner"])
            ypos += 1
        separators.append(ypos - 0.5)
        ypos += 1

    plt.figure(figsize=(11.6, 6.1))
    bars = plt.barh(y_values, score_values, color=["0.70" if o == "XYZ" else "0.35"
                                                   for o in owner_labels])
    for bar, score in zip(bars, score_values):
        plt.text(min(score + 0.017, 1.01), bar.get_y() + bar.get_height()/2,
                 f"{score:.4f}", va="center", fontsize=9)
    for sep in separators[:-1]:
        plt.axhline(sep + 0.5, lw=0.65, color="0.7")
    plt.yticks(y_values, row_labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1.12)
    plt.xlabel("Cosine similarity")
    plt.title("Why filtering the global Top-3 can miss an authorized result")
    plt.grid(axis="x", alpha=0.2)
    save("08_filter_before_search.png")

    filter_rows = []
    for group_name, hits in groups:
        filter_rows.extend((group_name, rank, h["doc_id"], h["owner"], h["score"])
                           for rank, h in enumerate(hits, 1))
    write_csv(path / "filtering_comparison.csv",
              ["approach", "rank", "doc_id", "owner", "cosine"], filter_rows)

    # Figure 6 — S02 requires re-embedding after a text update.
    plt.figure(figsize=(7.4, 4.8))
    bars = plt.bar(["Before update", "After update"], [before_s02, after_s02],
                   color=["0.72", "0.35"], width=0.55)
    for bar, score in zip(bars, (before_s02, after_s02)):
        plt.text(bar.get_x() + bar.get_width()/2, score + 0.025,
                 f"{score:.4f}", ha="center", fontsize=11)
    plt.ylim(0, 1.08)
    plt.ylabel("Cosine similarity with original question")
    plt.title("S02: document edit requires new embedding and reindexing")
    plt.grid(axis="y", alpha=0.2)
    save("09_update_and_reindex.png")
    write_csv(path / "update_comparison.csv", ["stage", "doc_id", "cosine"],
              [("before", "S02", before_s02), ("after", "S02", after_s02)])

    # Figure 7 — storage baseline only, NOT measured memory usage / index overhead.
    counts = np.array([1_000, 10_000, 100_000, 1_000_000, 10_000_000],
                      dtype=np.int64)
    dimensions = 384
    bytes_per_coordinate = 4  # float32
    size_gb = counts * dimensions * bytes_per_coordinate / 1e9
    plt.figure(figsize=(8.2, 4.9))
    plt.plot(counts, size_gb, "o-", linewidth=1.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of vectors (log scale)")
    plt.ylabel("Raw vector storage in decimal GB (log scale)")
    plt.title("Raw storage baseline: 384 dimensions per vector")
    plt.grid(alpha=0.3, which="both")
    plt.annotate(f"{size_gb[-1]:.2f} GB", (counts[-1], size_gb[-1]),
                 xytext=(-13, -16), textcoords="offset points", ha="right")
    save("10_vector_memory_baseline.png")
    write_csv(path / "vector_memory_baseline.csv",
              ["vector_count", "dimensions", "bytes_per_coordinate", "decimal_gb"],
              [(int(n), dimensions, bytes_per_coordinate, float(gb))
               for n, gb in zip(counts, size_gb)])

    # Additional operational context — deletion changes the index population.
    plt.figure(figsize=(7.2, 4.6))
    bars = plt.bar(["Before S13 deletion", "After S13 deletion"],
                   [starting_count, final_count], color=["0.72", "0.35"], width=0.58)
    for bar, number in zip(bars, (starting_count, final_count)):
        plt.text(bar.get_x() + bar.get_width()/2, number + 0.12,
                 str(number), ha="center", fontsize=11)
    plt.ylim(0, starting_count + 2)
    plt.ylabel("Active vectors in index")
    plt.title("Logical deletion followed by index rebuild")
    plt.grid(axis="y", alpha=0.2)
    save("12_deletion_and_rebuild.png")


def demo(path: Path, backend: str, plots: bool = False) -> None:
    conn = open_store(path / "lab.sqlite3")
    try:
        reset_demo_store(conn)
        index = build_index(conn, backend)
        starting_count = index.ntotal
        print(f"Backend: {backend}, vectors: {index.ntotal}, dimensions: {D}")

        global_top3 = search_global(index, conn, QUERY, k=3)
        print_hits("Global Top 3", global_top3)

        # Deliberately WRONG for complete tenant-scoped top-K: educational contrast.
        naive = [hit for hit in global_top3 if hit["owner"] == "ABC"]
        print_hits("Naive global Top-3 then ABC filter (incomplete)", naive)

        # Restrict to authorized candidates FIRST and find the true tenant Top-3.
        authorized = search_authorized_exact(conn, QUERY, owner="ABC", k=3)
        print_hits("Owner ABC: filter before exact search", authorized)
        assert [hit["doc_id"] for hit in authorized] == ["S03", "S04", "S01"]
        assert len(naive) == 2

        if backend == "faiss":
            index_path = path / "energy_flat.faiss"
            faiss.write_index(index, str(index_path))
            # Read only trusted local FAISS indexes; FAISS does not verify integrity.
            restored = faiss.read_index(str(index_path))
            assert restored.ntotal == index.ntotal
            print(f"\nSaved and reloaded index: {index_path}")

        before_s02 = score_document(conn, "S02", QUERY)
        update_document(conn, "S02", "Solar batteries store electricity to power lamps after dark.")
        after_s02 = score_document(conn, "S02", QUERY)
        index = build_index(conn, backend)
        print_hits("After modifying S02 and rebuilding", search_global(index, conn, QUERY, 3))
        print(f"S02 score changed: {before_s02:.4f} -> {after_s02:.4f}")
        assert after_s02 > before_s02

        deactivate_document(conn, "S13")
        index = build_index(conn, backend)
        final_count = index.ntotal
        hits = search_global(index, conn, QUERY, 5)
        assert all(hit["doc_id"] != "S13" for hit in hits)
        assert final_count == starting_count - 1
        print_hits("After deactivating S13 and rebuilding", hits)
        print(f"Active index population: {starting_count} -> {final_count}")

        if backend == "faiss":
            faiss.write_index(index, str(path / "energy_flat.faiss"))

        if plots:
            plot_demo_results(path, global_top3, naive, authorized,
                              before_s02, after_s02, starting_count, final_count)
    finally:
        conn.close()


def benchmark(path: Path, plots: bool, n: int = 10_000, d: int = 64) -> None:
    # Local ANN recall vs exact Flat + batched timings (synthetic random vectors).
    if faiss is None:
        print("FAISS unavailable; ANN benchmark skipped. Install: pip install faiss-cpu")
        return
    if n < 100 or d < 1:
        raise ValueError("Use --bench-n >= 100 and --bench-d >= 1")
    path.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(1)

    rng = np.random.default_rng(2026)
    vectors = np.ascontiguousarray(rng.normal(size=(n, d)).astype(np.float32))
    faiss.normalize_L2(vectors)
    query_count = min(100, n)
    queries = np.ascontiguousarray(
        vectors[rng.choice(n, size=query_count, replace=False)] +
        0.09 * rng.normal(size=(query_count, d)).astype(np.float32))
    faiss.normalize_L2(queries)

    k = 10
    flat = faiss.IndexFlatIP(d)
    flat.add(vectors)
    _, exact_ids = flat.search(queries, k)

    hnsw = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    hnsw.add(vectors)

    clusters = min(64, max(4, n // 60))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, clusters,
                             faiss.METRIC_INNER_PRODUCT)
    # 50 points per cluster generally gives an adequate training set for this toy benchmark.
    ivf.train(vectors[:min(n, max(3000, 50 * clusters))])
    ivf.add(vectors)

    def measure(label: str, idx):
        idx.search(queries, k)  # warm up
        start = time.perf_counter()
        _, ids = idx.search(queries, k)
        total_ms = (time.perf_counter() - start) * 1000
        recall = sum(len(set(ids[i]) & set(exact_ids[i]))
                     for i in range(len(queries))) / (len(queries) * k)
        result = {"index": label, "vectors": n, "dimension": d,
                  "recall_at_10": float(recall),
                  "batch_mean_ms_per_query": total_ms / len(queries)}
        print(f"{label:18s}  recall@10={recall:.3f}  "
              f"batched mean ms/query={result['batch_mean_ms_per_query']:.5f}")
        return result

    results = [measure("Flat exact", flat)]
    for ef in (8, 16, 32, 64):
        hnsw.hnsw.efSearch = ef
        results.append(measure(f"HNSW ef={ef}", hnsw))
    for nprobe in (1, 4, 16, 32):
        ivf.nprobe = min(nprobe, clusters)
        results.append(measure(f"IVF nprobe={nprobe}", ivf))

    with (path / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9.6, 5.5))
        for family in ("Flat", "HNSW", "IVF"):
            family_rows = [row for row in results if row["index"].startswith(family)]
            plt.plot([row["batch_mean_ms_per_query"] for row in family_rows],
                     [row["recall_at_10"] for row in family_rows],
                     marker="o", label=family)
            for row in family_rows:
                plt.annotate(row["index"],
                             (row["batch_mean_ms_per_query"], row["recall_at_10"]),
                             fontsize=8, xytext=(5, 5), textcoords="offset points")
        plt.xlabel("Mean per-query time in a 100-query batch (ms)")
        plt.ylabel("ANN Recall@10 relative to exact Flat")
        plt.ylim(-0.03, 1.06)
        plt.legend()
        plt.grid(alpha=0.2)
        plt.title(f"Synthetic FAISS benchmark, n={n:,}, d={d} (not production latency)")
        plt.tight_layout(pad=0.7)
        plt.savefig(path / "11_faiss_recall_latency.png", dpi=180,
                    bbox_inches="tight", pad_inches=0.06)
        plt.close()
        print("Saved plot: 11_faiss_recall_latency.png")

        # Bonus chart: how tuning the number of visited neighbours/clusters
        # influences the quality of an approximate search.
        plt.figure(figsize=(8.4, 4.8))
        for family in ("HNSW", "IVF"):
            rows = [row for row in results if row["index"].startswith(family)]
            knobs = [int(row["index"].split("=")[-1]) for row in rows]
            plt.plot(knobs, [row["recall_at_10"] for row in rows],
                     "o-", label="efSearch (HNSW)" if family == "HNSW" else "nprobe (IVF)")
        plt.xlabel("Index-specific tuning parameter (different meanings)")
        plt.ylabel("ANN Recall@10 relative to exact Flat")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(alpha=0.2)
        plt.title("ANN recall as search effort increases (synthetic test)")
        plt.tight_layout(pad=0.7)
        plt.savefig(path / "13_ann_tuning_recall.png", dpi=180,
                    bbox_inches="tight", pad_inches=0.06)
        plt.close()
        print("Saved plot: 13_ann_tuning_recall.png")

    print("\nTiming = single-thread, batched mean per query (not p95 online latency).")
    print("Recall here compares ANN to exact Flat; it is NOT human-judged relevance.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("vector_medium_output"),
                        help="Demo output directory; this directory's lab DB is reset on each run")
    parser.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    parser.add_argument("--plots", action="store_true",
                        help="Generate article figures 08-10 + lifecycle, even without FAISS")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run FAISS Flat/HNSW/IVF benchmark (needs faiss-cpu)")
    parser.add_argument("--bench-n", type=int, default=10_000,
                        help="Number of random vectors for benchmark (default: 10000)")
    parser.add_argument("--bench-d", type=int, default=64,
                        help="Random embedding dimensions for benchmark (default: 64)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    backend = ("faiss" if faiss is not None else "numpy") if args.backend == "auto" else args.backend
    if backend == "faiss" and faiss is None:
        parser.error("FAISS is missing. Run: python -m pip install faiss-cpu")
    if backend == "numpy" and faiss is None:
        print("FAISS not installed: NumPy exact-search fallback is active.")

    demo(args.out, backend, plots=args.plots)
    if args.benchmark:
        benchmark(args.out, args.plots, n=args.bench_n, d=args.bench_d)
    print("\nOutput directory:", args.out.resolve())


if __name__ == "__main__":
    main()
