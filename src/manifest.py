import bisect
import json
import os
import sqlite3
from datetime import datetime, timezone



MANIFEST_VERSION = 1


def compute_window_layout(
    seq_len,
    window_size,
    stride,
    include_terminal_window=True,
    index_short_contigs=True,
    min_short_contig_len=50,
    pad_short_contigs=True,
):
    """
    Compute the exact windows emitted for one contig.

    This is the single source of truth for FAISS ID assignment. Keep this logic
    synchronized with ingestion.sliding_window_fasta(...): regular stride-grid
    windows come first, followed by one optional terminal window. Short contigs
    can be represented by one N-padded window when they pass the minimum length.
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if min_short_contig_len < 0:
        raise ValueError("min_short_contig_len must be >= 0")

    if seq_len >= window_size:
        regular_window_count = ((seq_len - window_size) // stride) + 1
        last_regular_start = (regular_window_count - 1) * stride
        last_regular_end = last_regular_start + window_size
        has_terminal_window = bool(include_terminal_window and last_regular_end < seq_len)
        n_windows = regular_window_count + (1 if has_terminal_window else 0)
        return {
            "n_windows": n_windows,
            "regular_window_count": regular_window_count,
            "has_terminal_window": has_terminal_window,
            "is_short_contig": False,
            "padded_short_contig": False,
        }

    # Contigs shorter than the index window cannot produce a full-length native
    # window. When enabled, represent sufficiently informative short contigs by
    # one N-padded window. The embedder masks N, so padded bases do not add signal.
    should_index_short = (
        index_short_contigs
        and pad_short_contigs
        and seq_len >= min_short_contig_len
    )
    if should_index_short:
        return {
            "n_windows": 1,
            "regular_window_count": 0,
            "has_terminal_window": False,
            "is_short_contig": True,
            "padded_short_contig": True,
        }

    return {
        "n_windows": 0,
        "regular_window_count": 0,
        "has_terminal_window": False,
        "is_short_contig": False,
        "padded_short_contig": False,
    }


def _remove_sqlite_files(db_path):
    for suffix in ("", "-wal", "-shm"):
        path = db_path + suffix
        if os.path.exists(path):
            os.remove(path)


def _init_manifest_schema(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")

    cur.execute("""
        CREATE TABLE build_config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE fasta_files (
            fasta_id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            basename TEXT,
            size_bytes INTEGER,
            checksum TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE contigs (
            contig_id INTEGER PRIMARY KEY,
            fasta_id INTEGER NOT NULL,
            header TEXT NOT NULL,
            contig_length INTEGER NOT NULL,
            first_faiss_id INTEGER NOT NULL,
            last_faiss_id_exclusive INTEGER NOT NULL,
            n_windows INTEGER NOT NULL,
            regular_window_count INTEGER NOT NULL,
            has_terminal_window INTEGER NOT NULL,
            is_short_contig INTEGER NOT NULL,
            padded_short_contig INTEGER NOT NULL,
            stride INTEGER NOT NULL,
            window_size INTEGER NOT NULL,
            shard_id INTEGER NOT NULL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE INDEX contigs_faiss_range_idx
        ON contigs(first_faiss_id, last_faiss_id_exclusive)
    """)
    cur.execute("""
        CREATE TABLE shards (
            shard_id INTEGER PRIMARY KEY,
            first_faiss_id INTEGER NOT NULL,
            last_faiss_id_exclusive INTEGER NOT NULL,
            n_windows INTEGER NOT NULL,
            status TEXT NOT NULL,
            index_path TEXT,
            checksum TEXT,
            ntotal_expected INTEGER,
            ntotal_observed INTEGER,
            started_at TEXT,
            completed_at TEXT,
            error_message TEXT
        )
    """)
    conn.commit()


def _set_config(conn, values):
    cur = conn.cursor()
    records = [(key, json.dumps(value)) for key, value in values.items()]
    cur.executemany(
        "INSERT OR REPLACE INTO build_config (key, value) VALUES (?, ?)",
        records,
    )
    conn.commit()


def update_manifest_config(db_path, **values):
    """Update build_config entries after later build decisions such as nlist."""
    with sqlite3.connect(db_path) as conn:
        _set_config(conn, values)


def build_manifest(
    db_path,
    fasta_paths,
    window_size,
    stride,
    include_terminal_window=True,
    index_short_contigs=True,
    min_short_contig_len=50,
    pad_short_contigs=True,
    quantizer=None,
    pq_m=None,
    overwrite=True,
):
    """
    Build compact metadata/manifest SQLite DB.

    The manifest stores one row per contig, not one row per indexed window.
    FAISS IDs are assigned deterministically in FASTA order. Position recovery is
    done later from the contig ID range plus regular/terminal/short-contig flags.
    """
    import pyfastx

    if isinstance(fasta_paths, (str, bytes, os.PathLike)):
        fasta_paths = [str(fasta_paths)]
    else:
        fasta_paths = [str(path) for path in fasta_paths]

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    if overwrite:
        _remove_sqlite_files(db_path)

    conn = sqlite3.connect(db_path)
    try:
        _init_manifest_schema(conn)
        created_at = datetime.now(timezone.utc).isoformat()
        _set_config(conn, {
            "manifest_version": MANIFEST_VERSION,
            "created_at": created_at,
            "input_fasta_files": fasta_paths,
            "window_size": window_size,
            "db_stride": stride,
            "include_terminal_window": include_terminal_window,
            "index_short_contigs": index_short_contigs,
            "min_short_contig_len": min_short_contig_len,
            "pad_short_contigs": pad_short_contigs,
            "quantizer": quantizer,
            "pq_m": pq_m,
        })

        cur = conn.cursor()
        contig_id = 0
        total_windows = 0
        total_bases = 0
        total_indexed_contigs = 0

        for fasta_id, fasta_path in enumerate(fasta_paths):
            size_bytes = os.path.getsize(fasta_path) if os.path.exists(fasta_path) else None
            cur.execute(
                """
                INSERT INTO fasta_files (fasta_id, path, basename, size_bytes, checksum)
                VALUES (?, ?, ?, ?, ?)
                """,
                (fasta_id, fasta_path, os.path.basename(fasta_path), size_bytes, None),
            )

            for header, seq in pyfastx.Fasta(fasta_path, build_index=False):
                seq_len = len(seq)
                total_bases += seq_len
                layout = compute_window_layout(
                    seq_len,
                    window_size,
                    stride,
                    include_terminal_window=include_terminal_window,
                    index_short_contigs=index_short_contigs,
                    min_short_contig_len=min_short_contig_len,
                    pad_short_contigs=pad_short_contigs,
                )

                first_id = total_windows
                last_id = first_id + layout["n_windows"]
                if layout["n_windows"] > 0:
                    total_indexed_contigs += 1

                cur.execute(
                    """
                    INSERT INTO contigs (
                        contig_id, fasta_id, header, contig_length,
                        first_faiss_id, last_faiss_id_exclusive, n_windows,
                        regular_window_count, has_terminal_window,
                        is_short_contig, padded_short_contig,
                        stride, window_size, shard_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contig_id,
                        fasta_id,
                        header,
                        seq_len,
                        first_id,
                        last_id,
                        layout["n_windows"],
                        layout["regular_window_count"],
                        int(layout["has_terminal_window"]),
                        int(layout["is_short_contig"]),
                        int(layout["padded_short_contig"]),
                        stride,
                        window_size,
                        0,
                    ),
                )
                total_windows = last_id
                contig_id += 1

        cur.execute(
            """
            INSERT INTO shards (
                shard_id, first_faiss_id, last_faiss_id_exclusive, n_windows,
                status, index_path, checksum, ntotal_expected, ntotal_observed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (0, 0, total_windows, total_windows, "manifest_only", None, None, total_windows, None),
        )
        conn.commit()

        summary = {
            "db_path": db_path,
            "manifest_version": MANIFEST_VERSION,
            "input_fasta_files": fasta_paths,
            "total_contigs": contig_id,
            "total_indexed_contigs": total_indexed_contigs,
            "total_bases": total_bases,
            "total_windows": total_windows,
            "window_size": window_size,
            "stride": stride,
            "include_terminal_window": include_terminal_window,
            "index_short_contigs": index_short_contigs,
            "min_short_contig_len": min_short_contig_len,
            "pad_short_contigs": pad_short_contigs,
        }
        _set_config(conn, {
            "total_contigs": contig_id,
            "total_indexed_contigs": total_indexed_contigs,
            "total_bases": total_bases,
            "total_windows": total_windows,
        })
        return summary
    finally:
        conn.close()


class CompactMetadataLookup:
    """
    In-memory resolver for compact manifest metadata.

    It replaces per-window SQLite lookups. A FAISS ID is mapped to a contig row
    by binary searching the compact ID ranges, then start/end coordinates are
    reconstructed from the local ordinal and the stored window layout fields.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.rows = []
        self.first_ids = []
        self._load()

    def _load(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT contig_id, header, contig_length,
                       first_faiss_id, last_faiss_id_exclusive, n_windows,
                       regular_window_count, has_terminal_window,
                       is_short_contig, padded_short_contig,
                       stride, window_size, shard_id
                FROM contigs
                WHERE n_windows > 0
                ORDER BY first_faiss_id
                """
            )
            self.rows = [dict(row) for row in cur.fetchall()]

        if not self.rows:
            raise ValueError(f"No compact contig metadata found in {self.db_path}")
        self.first_ids = [row["first_faiss_id"] for row in self.rows]

    def _row_for_id(self, faiss_id):
        pos = bisect.bisect_right(self.first_ids, faiss_id) - 1
        if pos < 0:
            return None
        row = self.rows[pos]
        if faiss_id >= row["last_faiss_id_exclusive"]:
            return None
        return row

    @staticmethod
    def _position_from_row(row, faiss_id):
        local_ordinal = faiss_id - row["first_faiss_id"]

        if row["is_short_contig"]:
            return 0, row["contig_length"]

        if local_ordinal < row["regular_window_count"]:
            start_pos = local_ordinal * row["stride"]
            return start_pos, start_pos + row["window_size"]

        # The only valid non-regular ordinal is the optional terminal window.
        if row["has_terminal_window"]:
            start_pos = row["contig_length"] - row["window_size"]
            return start_pos, row["contig_length"]

        return None, None

    def fetch_metadata_batch(self, faiss_ids):
        result = {}
        for raw_id in faiss_ids:
            faiss_id = int(raw_id)
            row = self._row_for_id(faiss_id)
            if row is None:
                continue
            start_pos, end_pos = self._position_from_row(row, faiss_id)
            if start_pos is None:
                continue
            result[faiss_id] = {
                "header": row["header"],
                "start_pos": int(start_pos),
                "end_pos": int(end_pos),
                "contig_id": int(row["contig_id"]),
                "shard_id": int(row["shard_id"]),
            }
        return result
