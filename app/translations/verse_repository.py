
import sqlite3
from pathlib import Path


class VerseRepository:
    def __init__(self, db_path: str = "data/bible.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS verses (
            translation_id TEXT NOT NULL,
            verse_id TEXT NOT NULL,
            book_id TEXT NOT NULL,
            book_order INTEGER NOT NULL,
            book_name TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL,

            PRIMARY KEY (translation_id, verse_id)
        )
        """

        with self.connect() as conn:
            conn.execute(query)
            conn.commit()

    def insert_verses(self, records: list[dict]):
        query = """
        INSERT OR REPLACE INTO verses (
            translation_id, verse_id, book_id, book_order,
            book_name, chapter, verse, text, source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = [
            (
                record["translation_id"],
                record["verse_id"],
                record["book_id"],
                record["book_order"],
                record["book_name"],
                record["chapter"],
                record["verse"],
                record["text"],
                record["source"],
            )
            for record in records
        ]

        with self.connect() as conn:
            conn.executemany(query, values)
            conn.commit()

    def get_verse(
        self,
        translation_id: str,
        book_id: str,
        chapter: int,
        verse: int,
    ) -> dict | None:
        query = """
        SELECT *
        FROM verses
        WHERE translation_id = ?
          AND book_id = ?
          AND chapter = ?
          AND verse = ?
        """

        with self.connect() as conn:
            cursor = conn.execute(
                query,
                (translation_id, book_id, chapter, verse),
            )
            row = cursor.fetchone()

        return dict(row) if row else None

    def get_chapter(
        self,
        translation_id: str,
        book_id: str,
        chapter: int,
    ) -> list[dict]:
        query = """
        SELECT *
        FROM verses
        WHERE translation_id = ?
          AND book_id = ?
          AND chapter = ?
        ORDER BY verse
        """

        with self.connect() as conn:
            cursor = conn.execute(
                query,
                (translation_id, book_id, chapter),
            )
            rows = cursor.fetchall()

        return [dict(row) for row in rows]
