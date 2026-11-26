"""Persistent SQLite storage for discrete optimizer evaluations."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .evaluator import EvaluationResult


class EvaluationStore:
    """Store one immutable-by-default result per geometry fingerprint.

    Existing records (including failed runs) are never silently replaced.
    Pass replace=True to save_result only when an explicit retry is intended.
    """

    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        if self.database != ":memory:":
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                fingerprint TEXT PRIMARY KEY,
                indices_json TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('ok', 'failed')),
                objective REAL NOT NULL,
                evaluation_dir TEXT NOT NULL,
                elapsed_seconds REAL NOT NULL,
                metrics_json TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_status_objective "
            "ON evaluations(status, objective)"
        )
        self._connection.commit()

    @staticmethod
    def _serialize(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> EvaluationResult:
        return EvaluationResult(
            fingerprint=row["fingerprint"],
            indices=tuple(json.loads(row["indices_json"])),
            parameters=json.loads(row["parameters_json"]),
            status=row["status"],
            objective=float(row["objective"]),
            evaluation_dir=row["evaluation_dir"],
            elapsed_seconds=float(row["elapsed_seconds"]),
            metrics=json.loads(row["metrics_json"]),
            error=row["error"],
        )

    def save_result(self, result: EvaluationResult, replace: bool = False) -> bool:
        """Persist a result; return False if it already exists and replace is False."""
        if result.status not in ("ok", "failed"):
            raise ValueError("result.status must be 'ok' or 'failed'")
        statement = (
            """
            INSERT INTO evaluations (
                fingerprint, indices_json, parameters_json, status, objective,
                evaluation_dir, elapsed_seconds, metrics_json, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        values = (
            result.fingerprint,
            self._serialize(list(result.indices)),
            self._serialize(result.parameters),
            result.status,
            float(result.objective),
            result.evaluation_dir,
            float(result.elapsed_seconds),
            self._serialize(result.metrics),
            result.error,
        )
        if replace:
            statement += """
            ON CONFLICT(fingerprint) DO UPDATE SET
                indices_json=excluded.indices_json,
                parameters_json=excluded.parameters_json,
                status=excluded.status,
                objective=excluded.objective,
                evaluation_dir=excluded.evaluation_dir,
                elapsed_seconds=excluded.elapsed_seconds,
                metrics_json=excluded.metrics_json,
                error=excluded.error,
                created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
            """
        else:
            statement += " ON CONFLICT(fingerprint) DO NOTHING"
        cursor = self._connection.execute(statement, values)
        self._connection.commit()
        return cursor.rowcount == 1

    def get(self, fingerprint: str) -> EvaluationResult | None:
        row = self._connection.execute(
            "SELECT * FROM evaluations WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return None if row is None else self._from_row(row)

    def contains(self, fingerprint: str) -> bool:
        return self._connection.execute(
            "SELECT 1 FROM evaluations WHERE fingerprint = ?", (fingerprint,)
        ).fetchone() is not None

    def list_results(self, status: str | None = None) -> list[EvaluationResult]:
        if status is not None and status not in ("ok", "failed"):
            raise ValueError("status must be 'ok', 'failed', or None")
        if status is None:
            rows = self._connection.execute(
                "SELECT * FROM evaluations ORDER BY rowid"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM evaluations WHERE status = ? "
                "ORDER BY rowid", (status,)
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def best(self) -> EvaluationResult | None:
        """Return the lowest-cost successful evaluation, or None."""
        row = self._connection.execute(
            "SELECT * FROM evaluations WHERE status = 'ok' "
            "ORDER BY objective ASC, created_at ASC LIMIT 1"
        ).fetchone()
        return None if row is None else self._from_row(row)

    def count(self, status: str | None = None) -> int:
        if status is not None and status not in ("ok", "failed"):
            raise ValueError("status must be 'ok', 'failed', or None")
        if status is None:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM evaluations"
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM evaluations WHERE status = ?", (status,)
            ).fetchone()
        return int(row["n"])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "EvaluationStore":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
