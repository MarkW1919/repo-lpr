#!/usr/bin/env bash
# RepoScan Pro - Backup & Restore Script
set -euo pipefail

BACKUP_DIR="/data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_CONTAINER="reposcan-db-1"

usage() {
    echo "Usage: $0 {backup|restore <file>}"
    echo ""
    echo "  backup              Create a backup of the database and config"
    echo "  restore <file>      Restore from a backup file"
    exit 1
}

backup() {
    echo "Creating backup..."
    mkdir -p "$BACKUP_DIR"

    BACKUP_FILE="$BACKUP_DIR/reposcan_$TIMESTAMP.tar.gz"

    # Dump database (pipefail ensures pg_dump errors are caught, not masked by gzip)
    echo "  Dumping database..."
    docker exec "$DB_CONTAINER" pg_dump -U reposcan reposcan | gzip > "$BACKUP_DIR/db_$TIMESTAMP.sql.gz"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "ERROR: pg_dump failed"
        rm -f "$BACKUP_DIR/db_$TIMESTAMP.sql.gz"
        exit 1
    fi

    # Package everything
    echo "  Packaging backup..."
    tar czf "$BACKUP_FILE" \
        -C "$BACKUP_DIR" "db_$TIMESTAMP.sql.gz" \
        -C /opt/reposcan .env \
        2>/dev/null || true

    rm -f "$BACKUP_DIR/db_$TIMESTAMP.sql.gz"

    echo "Backup saved to: $BACKUP_FILE"
    echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"
    chmod 600 "$BACKUP_FILE"
}

restore() {
    local FILE="$1"

    if [ ! -f "$FILE" ]; then
        echo "Error: Backup file not found: $FILE"
        exit 1
    fi

    echo "Restoring from: $FILE"
    echo "WARNING: This will overwrite the current database!"
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi

    TMP_DIR=$(mktemp -d)
    tar xzf "$FILE" -C "$TMP_DIR"

    # Find and restore database dump
    DB_DUMP=$(find "$TMP_DIR" -name "db_*.sql.gz" | head -1)
    if [ -n "$DB_DUMP" ]; then
        echo "  Restoring database..."
        gunzip -c "$DB_DUMP" | docker exec -i "$DB_CONTAINER" psql -U reposcan reposcan
    fi

    rm -rf "$TMP_DIR"
    echo "Restore complete."
}

case "${1:-}" in
    backup)
        backup
        ;;
    restore)
        if [ -z "${2:-}" ]; then
            echo "Error: Specify a backup file to restore"
            usage
        fi
        restore "$2"
        ;;
    *)
        usage
        ;;
esac
