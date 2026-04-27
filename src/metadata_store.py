import sqlite3
import os

class MetadataStore:
    def __init__(self, db_path):
        """
        Initializes the disk-backed SQLite database.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates the table and optimizes the database for fast bulk inserts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = None
            try:
                cursor = conn.cursor()
                # WAL mode drastically improves write speeds
                cursor.execute("PRAGMA journal_mode = WAL;")
                # Synchronous = NORMAL is safe in WAL mode and much faster
                cursor.execute("PRAGMA synchronous = NORMAL;")
                
                # The 'id' is the FAISS 64-bit integer.
                # Making it the PRIMARY KEY automatically builds a B-Tree index for instant retrieval.
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        id INTEGER PRIMARY KEY,
                        header TEXT,
                        start_pos INTEGER,
                        end_pos INTEGER
                    )
                """)
            finally:
                if cursor:
                    cursor.close()

    def insert_batch(self, start_id, metadata_batch):
        """
        Inserts a batch of metadata records using a single transaction.
        Returns the next available ID.
        """
        # Prepare the data for SQLite: (id, header, start_pos, end_pos)
        records = []
        current_id = start_id
        for header, start_pos, end_pos in metadata_batch:
            records.append((current_id, header, start_pos, end_pos))
            current_id += 1

        with sqlite3.connect(self.db_path) as conn:
            cursor = None
            try:
                cursor = conn.cursor()
                cursor.executemany(
                    "INSERT INTO metadata (id, header, start_pos, end_pos) VALUES (?, ?, ?, ?)",
                    records
                )
            finally:
                if cursor:
                    cursor.close()
        return current_id

    def fetch_metadata(self, faiss_id):
        """
        Retrieves the biological metadata for a given FAISS ID.
        Used later in Phase 2 (The Mapper).
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT header, start_pos, end_pos FROM metadata WHERE id = ?", (faiss_id,))
            return cursor.fetchone()

    def fetch_metadata_batch(self, faiss_ids):
        """Optional: Retrieves thousands of records instantly for the Mapper."""
        ids_str = ",".join(map(str, faiss_ids))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT id, header, start_pos, end_pos FROM metadata WHERE id IN ({ids_str})")
            results = cursor.fetchall()
            # Returns a dictionary: {id: (header, start_pos, end_pos)}
            return {row[0]: (row[1], row[2], row[3]) for row in results}