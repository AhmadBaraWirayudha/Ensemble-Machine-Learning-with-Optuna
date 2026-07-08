# Runtime data

- `production.db` - SQLite database with every prediction served, every
  physical measurement submitted, and every auto-retrain attempt. Created
  on first use. See `monitoring/storage.py` for the schema, and
  `scripts/migrate_jsonl_to_sqlite.py` if you're upgrading from an older
  version of this project that used flat `.jsonl` files here instead.

Not committed to git (see `.gitignore`); the `.gitkeep`/`README.md` files
here just preserve the directory structure.
