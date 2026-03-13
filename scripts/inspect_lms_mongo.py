"""Inspect LMS MongoDB collections and sample document shapes.

Usage:
    uv run python scripts/inspect_lms_mongo.py

Environment:
    MONGODB_URI - Mongo connection string (required)
    MONGO_SAMPLE_SIZE - Number of sample docs per collection (default: 2)
    MONGO_SCHEMA_OUT - Output JSON path for schema map (default: scripts/mongo_schema_map.json)
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except ImportError:
        return


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def _shape(value: Any, depth: int = 0, max_depth: int = 2) -> Any:
    if depth >= max_depth:
        return _type_name(value)

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, val in value.items():
            out[str(key)] = _shape(val, depth + 1, max_depth)
        return out

    if isinstance(value, list):
        if not value:
            return []
        sample = value[0]
        return [_shape(sample, depth + 1, max_depth)]

    return _type_name(value)


def _preview_document(doc: dict[str, Any], max_len: int = 800) -> str:
    rendered = json.dumps(doc, default=str, indent=2)
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "\n... (truncated)"


def _shape_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _shape_signature(v) for k, v in value.items()}
    if isinstance(value, list):
        if not value:
            return ["empty"]
        return [_shape_signature(value[0])]
    return str(value)


def _merge_schema(into: dict[str, Any], sample: dict[str, Any]) -> None:
    for key, val in sample.items():
        if key not in into:
            into[key] = val
            continue

        existing = into[key]
        if isinstance(existing, dict) and isinstance(val, dict):
            _merge_schema(existing, val)
            continue

        if isinstance(existing, list) and isinstance(val, list):
            if not existing:
                into[key] = val
                continue
            if not val:
                continue
            if isinstance(existing[0], dict) and isinstance(val[0], dict):
                _merge_schema(existing[0], val[0])
            elif existing[0] != val[0]:
                merged: list[Any] = [existing[0]]
                if val[0] not in merged:
                    merged.append(val[0])
                into[key] = merged
            continue

        if existing != val:
            merged_types: list[str] = []
            for item in (existing, val):
                text = str(item)
                if text not in merged_types:
                    merged_types.append(text)
            into[key] = merged_types


def _default_schema_output_path() -> str:
    return os.path.join(os.path.dirname(__file__), "mongo_schema_map.json")


def main() -> int:
    _maybe_load_dotenv()

    mongo_uri = os.getenv("MONGODB_URI", "").strip()
    if not mongo_uri:
        print("ERROR: MONGODB_URI is not set")
        return 1

    sample_size_raw = os.getenv("MONGO_SAMPLE_SIZE", "2").strip() or "2"
    try:
        sample_size = max(1, int(sample_size_raw))
    except ValueError:
        sample_size = 2

    schema_out = os.getenv("MONGO_SCHEMA_OUT", "").strip() or _default_schema_output_path()

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")

        db = client.get_default_database()
        if db is None:
            print("ERROR: No default database found in MONGODB_URI")
            return 1

        print("=" * 72)
        print("LMS MongoDB Inspection")
        print("=" * 72)
        print(f"Database: {db.name}")

        collections = sorted(db.list_collection_names())
        print(f"Collections: {len(collections)}")

        schema_map: dict[str, Any] = {
            "database": db.name,
            "sample_size": sample_size,
            "collections": {},
        }

        for name in collections:
            collection = db[name]
            try:
                count = collection.estimated_document_count()
            except Exception:
                count = -1

            print("\n" + "-" * 72)
            print(f"Collection: {name}")
            print(f"Estimated docs: {count if count >= 0 else 'unknown'}")

            docs = list(collection.find({}, limit=sample_size))
            if not docs:
                print("Sample shape: <empty collection>")
                schema_map["collections"][name] = {
                    "estimated_docs": count if count >= 0 else None,
                    "shape": None,
                    "shape_signature": None,
                }
                continue

            first = docs[0]
            merged_shape: dict[str, Any] = {}
            for doc in docs:
                doc_shape = _shape(doc)
                if isinstance(doc_shape, dict):
                    _merge_schema(merged_shape, doc_shape)

            print("Sample shape (from first document):")
            print(json.dumps(_shape(first), indent=2, default=str)[:1500])

            print("Sample document preview:")
            print(_preview_document(first))

            schema_map["collections"][name] = {
                "estimated_docs": count if count >= 0 else None,
                "shape": merged_shape,
                "shape_signature": _shape_signature(merged_shape),
            }

        out_dir = os.path.dirname(schema_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(schema_out, "w", encoding="utf-8") as handle:
            json.dump(schema_map, handle, indent=2, default=str)
            handle.write("\n")

        print(f"\nSchema map written to: {schema_out}")

        print("\nInspection complete.")
        return 0

    except PyMongoError as exc:
        print(f"Mongo error: {exc}")
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
