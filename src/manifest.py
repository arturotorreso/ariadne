import bisect
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone


MANIFEST_VERSION = 2


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


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

    This is the single source of truth for FAISS ID assignment. Regular
    stride-grid windows come first. If the stride grid misses the contig end,
    one optional terminal full-length window is appended. Short contigs can be
    represented by one N-padded window when they pass the minimum length.
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
            shard_id INTEGER NOT NULL DEFAULT -1
        )
    """)
    cur.execute("""
        CREATE INDEX contigs_faiss_range_idx
        ON contigs(first_faiss_id, last_faiss_id_exclusive)
    """)
    cur.execute("""
        CREATE INDEX contigs_shard_idx
        ON contigs(shard_id, contig_id)
    """)
    cur.execute("""
        CREATE TABLE shards (
            shard_id INTEGER PRIMARY KEY,
            first_faiss_id INTEGER NOT NULL,
            last_faiss_id_exclusive INTEGER NOT NULL,
            n_windows INTEGER NOT NULL,
            n_contigs INTEGER NOT NULL,
            status TEXT NOT NULL,
            index_path TEXT,
            tmp_index_path TEXT,
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
    records = [(key, json.dumps(value)) for key, value in values.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO build_config (key, value) VALUES (?, ?)",
        records,
    )
    conn.commit()


def update_manifest_config(db_path, **values):
    """Update build_config entries after later build decisions such as nlist."""
    with sqlite3.connect(db_path) as conn:
        _set_config(conn, values)


def get_manifest_config(db_path):
    """Return build_config as a Python dictionary with JSON-decoded values."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM build_config")
        return {key: json.loads(value) for key, value in cur.fetchall()}


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
        created_at = utc_now_iso()
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
                        -1,
                    ),
                )
                total_windows = last_id
                contig_id += 1

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


def get_manifest_summary(db_path):
    config = get_manifest_config(db_path)
    return {
        "db_path": db_path,
        "manifest_version": config.get("manifest_version"),
        "input_fasta_files": config.get("input_fasta_files"),
        "total_contigs": config.get("total_contigs", 0),
        "total_indexed_contigs": config.get("total_indexed_contigs", 0),
        "total_bases": config.get("total_bases", 0),
        "total_windows": config.get("total_windows", 0),
        "window_size": config.get("window_size"),
        "stride": config.get("db_stride"),
        "include_terminal_window": config.get("include_terminal_window"),
        "index_short_contigs": config.get("index_short_contigs"),
        "min_short_contig_len": config.get("min_short_contig_len"),
        "pad_short_contigs": config.get("pad_short_contigs"),
    }


def file_sha256(path, block_size=8 * 1024 * 1024):
    """Compute a SHA256 checksum without loading the whole index file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_shards(db_path, shards_dir, target_shard_windows=10_000_000, overwrite=True):
    """
    Assign indexed contigs to shard work units.

    First implementation uses contig-boundary shards: a contig is never split.
    If a single contig exceeds target_shard_windows, it becomes its own shard.
    """
    if target_shard_windows <= 0:
        raise ValueError("target_shard_windows must be > 0")

    os.makedirs(shards_dir, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if overwrite:
            cur.execute("DELETE FROM shards")
            cur.execute("UPDATE contigs SET shard_id = -1")

        cur.execute(
            """
            SELECT contig_id, first_faiss_id, last_faiss_id_exclusive, n_windows
            FROM contigs
            WHERE n_windows > 0
            ORDER BY contig_id
            """
        )
        contigs = [dict(row) for row in cur.fetchall()]
        if not contigs:
            raise ValueError("Manifest contains no indexed windows; cannot assign shards.")

        shards = []
        current = []
        current_windows = 0

        def flush_current():
            nonlocal current, current_windows
            if not current:
                return
            shard_id = len(shards)
            first_id = current[0]["first_faiss_id"]
            last_id = current[-1]["last_faiss_id_exclusive"]
            shards.append({
                "shard_id": shard_id,
                "contigs": current,
                "first_faiss_id": first_id,
                "last_faiss_id_exclusive": last_id,
                "n_windows": current_windows,
                "n_contigs": len(current),
            })
            current = []
            current_windows = 0

        for row in contigs:
            row_windows = int(row["n_windows"])
            if current and current_windows + row_windows > target_shard_windows:
                flush_current()
            current.append(row)
            current_windows += row_windows
            if row_windows >= target_shard_windows:
                flush_current()
        flush_current()

        for shard in shards:
            shard_id = shard["shard_id"]
            index_path = os.path.join(shards_dir, f"shard_{shard_id:06d}.index")
            tmp_index_path = index_path + ".tmp"
            cur.execute(
                """
                INSERT INTO shards (
                    shard_id, first_faiss_id, last_faiss_id_exclusive,
                    n_windows, n_contigs, status, index_path, tmp_index_path,
                    checksum, ntotal_expected, ntotal_observed,
                    started_at, completed_at, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shard_id,
                    shard["first_faiss_id"],
                    shard["last_faiss_id_exclusive"],
                    shard["n_windows"],
                    shard["n_contigs"],
                    "pending",
                    index_path,
                    tmp_index_path,
                    None,
                    shard["n_windows"],
                    None,
                    None,
                    None,
                    None,
                ),
            )
            contig_ids = [row["contig_id"] for row in shard["contigs"]]
            placeholders = ",".join("?" for _ in contig_ids)
            cur.execute(
                f"UPDATE contigs SET shard_id = ? WHERE contig_id IN ({placeholders})",
                [shard_id] + contig_ids,
            )

        _set_config(conn, {
            "target_shard_windows": target_shard_windows,
            "num_shards": len(shards),
            "shards_dir": shards_dir,
        })
        conn.commit()
        return {
            "num_shards": len(shards),
            "target_shard_windows": target_shard_windows,
            "shards_dir": shards_dir,
            "total_windows": sum(shard["n_windows"] for shard in shards),
        }


def get_shards(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT shard_id, first_faiss_id, last_faiss_id_exclusive,
                   n_windows, n_contigs, status, index_path, tmp_index_path,
                   checksum, ntotal_expected, ntotal_observed,
                   started_at, completed_at, error_message
            FROM shards
            ORDER BY shard_id
            """
        )
        return [dict(row) for row in cur.fetchall()]


def get_shard(db_path, shard_id):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT shard_id, first_faiss_id, last_faiss_id_exclusive,
                   n_windows, n_contigs, status, index_path, tmp_index_path,
                   checksum, ntotal_expected, ntotal_observed,
                   started_at, completed_at, error_message
            FROM shards
            WHERE shard_id = ?
            """,
            (shard_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Shard {shard_id} does not exist in {db_path}")
        return dict(row)


def update_shard_status(db_path, shard_id, status, **fields):
    fields = dict(fields)
    fields["status"] = status
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [shard_id]
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"UPDATE shards SET {assignments} WHERE shard_id = ?", values)
        conn.commit()


def validate_shard_index(index_path, expected_ntotal):
    """
    Return (is_valid, observed_ntotal, error_message) for a saved shard index.
    FAISS is imported lazily so manifest-only operations do not require it.
    """
    if not index_path or not os.path.exists(index_path):
        return False, None, "index file does not exist"
    try:
        import faiss
        index = faiss.read_index(index_path)
        observed = int(index.ntotal)
        if observed != int(expected_ntotal):
            return False, observed, f"ntotal mismatch: observed {observed}, expected {expected_ntotal}"
        return True, observed, None
    except Exception as exc:
        return False, None, str(exc)


def _load_contig_rows_for_stream(db_path, shard_id=None):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT fasta_id, path FROM fasta_files ORDER BY fasta_id")
        fasta_files = [dict(row) for row in cur.fetchall()]

        query = """
            SELECT contig_id, fasta_id, header, contig_length,
                   first_faiss_id, last_faiss_id_exclusive, n_windows,
                   regular_window_count, has_terminal_window,
                   is_short_contig, padded_short_contig,
                   stride, window_size, shard_id
            FROM contigs
            WHERE n_windows > 0
        """
        params = []
        if shard_id is not None:
            query += " AND shard_id = ?"
            params.append(shard_id)
        query += " ORDER BY contig_id"
        cur.execute(query, params)
        rows = [dict(row) for row in cur.fetchall()]
    return fasta_files, {row["contig_id"]: row for row in rows}


def stream_windows_from_manifest(db_path, batch_size, shard_id=None):
    """
    Stream sequence windows and global FAISS IDs from the compact manifest.

    This generator is used by both global training and shard construction. It
    guarantees that emitted windows match manifest ID ranges, including regular
    windows, optional terminal windows, and padded short-contig windows.
    """
    import pyfastx

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    fasta_files, selected_rows = _load_contig_rows_for_stream(db_path, shard_id=shard_id)
    seq_batch = []
    id_batch = []
    contig_id = 0

    def append_window(window_seq, faiss_id):
        seq_batch.append(str(window_seq).upper())
        id_batch.append(int(faiss_id))
        if len(seq_batch) == batch_size:
            out_seq = list(seq_batch)
            out_ids = list(id_batch)
            seq_batch.clear()
            id_batch.clear()
            return out_seq, out_ids
        return None

    for fasta in fasta_files:
        for header, seq in pyfastx.Fasta(fasta["path"], build_index=False):
            row = selected_rows.get(contig_id)
            if row is not None:
                seq_len = int(row["contig_length"])
                window_size = int(row["window_size"])
                stride = int(row["stride"])
                first_id = int(row["first_faiss_id"])

                if row["is_short_contig"]:
                    # Keep true coordinates in metadata, but embed a full-length
                    # padded sequence. N bases are masked by the embedder.
                    short_seq = str(seq[0:seq_len]).upper().ljust(window_size, "N")
                    yielded = append_window(short_seq, first_id)
                    if yielded is not None:
                        yield yielded
                else:
                    regular_count = int(row["regular_window_count"])
                    for local_ordinal in range(regular_count):
                        start = local_ordinal * stride
                        end = start + window_size
                        yielded = append_window(seq[start:end], first_id + local_ordinal)
                        if yielded is not None:
                            yield yielded

                    if row["has_terminal_window"]:
                        terminal_start = seq_len - window_size
                        terminal_id = first_id + regular_count
                        yielded = append_window(seq[terminal_start:seq_len], terminal_id)
                        if yielded is not None:
                            yield yielded
            contig_id += 1

    if seq_batch:
        yield list(seq_batch), list(id_batch)


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
