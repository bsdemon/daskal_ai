import sqlite3
import os
import json
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from src.core.config import settings


class ConfigDB:
    """SQLite-based configuration database."""

    def __init__(self, db_path: str = "") -> None:
        # Set DB path to a file in the ChromaDB directory for consistency
        if db_path is None:
            os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            db_path = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, "config.db")

        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create settings table for configuration values
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    description TEXT,
                    group_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create a trigger to update the updated_at timestamp
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_settings_timestamp
                AFTER UPDATE ON settings
                BEGIN
                    UPDATE settings SET updated_at = CURRENT_TIMESTAMP WHERE key = NEW.key;
                END
            """
            )

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with context management."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            # Return dict objects instead of tuples
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()

    def _convert_value(self, value: str, value_type: str) -> Any:
        """Convert a string value to the appropriate Python type."""
        if value_type == "str":
            return value
        elif value_type == "int":
            return int(value)
        elif value_type == "float":
            return float(value)
        elif value_type == "bool":
            return value.lower() in ("true", "1", "yes", "y")
        elif value_type == "json":
            return json.loads(value)
        else:
            return value

    def get_setting(self, key: str) -> Optional[Any]:
        """Get a setting value by key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value, value_type FROM settings WHERE key = ?", (key,)
            )
            result = cursor.fetchone()

            if result:
                return self._convert_value(result["value"], result["value_type"])
            return None

    def get_settings_by_group(self, group_name: str) -> Dict[str, Any]:
        """Get all settings for a specific group."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT key, value, value_type FROM settings WHERE group_name = ?",
                (group_name,),
            )
            results = cursor.fetchall()

            return {
                row["key"]: self._convert_value(row["value"], row["value_type"])
                for row in results
            }

    def get_all_settings(self) -> List[Dict[str, Any]]:
        """Get all settings with their metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT key, value, value_type, description, group_name, created_at, updated_at FROM settings"
            )
            results = cursor.fetchall()

            return [
                {
                    "key": row["key"],
                    "value": self._convert_value(row["value"], row["value_type"]),
                    "value_type": row["value_type"],
                    "description": row["description"],
                    "group_name": row["group_name"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in results
            ]

    def set_setting(
        self,
        key: str,
        value: Any,
        value_type: Optional[str] = None,
        description: Optional[str] = None,
        group_name: Optional[str] = None,
    ) -> None:
        """Set a setting value by key."""
        # Determine value type if not provided
        if value_type is None:
            if isinstance(value, str):
                value_type = "str"
            elif isinstance(value, int):
                value_type = "int"
            elif isinstance(value, float):
                value_type = "float"
            elif isinstance(value, bool):
                value_type = "bool"
            elif isinstance(value, (dict, list)):
                value_type = "json"
                value = json.dumps(value)
            else:
                value_type = "str"
                value = str(value)

        # Convert value to string for storage
        if value_type == "json" and not isinstance(value, str):
            string_value = json.dumps(value)
        elif value_type == "bool":
            string_value = str(value).lower()
        else:
            string_value = str(value)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if setting already exists
            cursor.execute("SELECT 1 FROM settings WHERE key = ?", (key,))
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing setting
                update_cols = ["value = ?", "value_type = ?"]
                params = [string_value, value_type]

                if description is not None:
                    update_cols.append("description = ?")
                    params.append(description)

                if group_name is not None:
                    update_cols.append("group_name = ?")
                    params.append(group_name)

                query = f"UPDATE settings SET {', '.join(update_cols)} WHERE key = ?"
                params.append(key)

                cursor.execute(query, params)
            else:
                # Insert new setting
                cursor.execute(
                    "INSERT INTO settings (key, value, value_type, description, group_name) VALUES (?, ?, ?, ?, ?)",
                    (key, string_value, value_type, description, group_name),
                )

            conn.commit()

    def delete_setting(self, key: str) -> bool:
        """Delete a setting by key. Returns True if a setting was deleted."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM settings WHERE key = ?", (key,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def initialize_default_settings(self) -> None:
        """Initialize the database with default settings from the config."""
        # Feature flags
        self.set_setting(
            "ENABLE_EMBEDDING",
            settings.ENABLE_EMBEDDING,
            "bool",
            "Enable embedding service",
            "features",
        )

        self.set_setting(
            "ENABLE_CONTEXTUAL_EMBEDDING",
            settings.ENABLE_CONTEXTUAL_EMBEDDING,
            "bool",
            "Enable contextual descriptions for embeddings",
            "features",
        )

        self.set_setting(
            "ENABLE_RERANKING",
            settings.ENABLE_RERANKING,
            "bool",
            "Enable reranking service",
            "features",
        )

        # RAG configuration
        self.set_setting(
            "CHUNK_SIZE",
            settings.CHUNK_SIZE,
            "int",
            "Size of text chunks for RAG",
            "rag",
        )

        self.set_setting(
            "CHUNK_OVERLAP",
            settings.CHUNK_OVERLAP,
            "int",
            "Overlap between text chunks",
            "rag",
        )

        self.set_setting(
            "MAX_CHUNKS",
            settings.MAX_CHUNKS,
            "int",
            "Maximum number of chunks to retrieve",
            "rag",
        )

        self.set_setting(
            "TEMPERATURE",
            settings.TEMPERATURE,
            "float",
            "Temperature for LLM generation",
            "rag",
        )

        # Default providers
        self.set_setting(
            "DEFAULT_LLM_PROVIDER",
            settings.DEFAULT_LLM_PROVIDER,
            "str",
            "Default LLM provider: anthropic, openai, or gemini",
            "providers",
        )

        self.set_setting(
            "DEFAULT_EMBEDDING_PROVIDER",
            settings.DEFAULT_EMBEDDING_PROVIDER,
            "str",
            "Default embedding provider",
            "providers",
        )

        self.set_setting(
            "DEFAULT_RERANKING_METHOD",
            settings.DEFAULT_RERANKING_METHOD,
            "str",
            "Default reranking method: bm25, cohere",
            "providers",
        )


# Create a global instance
config_db = ConfigDB()
