"""Command line: DB schema and external taxonomies.

analog-search db init
analog-search sources fetch ksr|okpd2|etim
analog-search sources register ksr|ktru FILE --version V [--published YYYY-MM-DD]
analog-search sources load ksr|okpd2|ktru|etim [--dir DIR] [--force]
analog-search sources status
analog-search knowledge load
analog-search positions load [FILE]
analog-search decisions load [FILE] [--author A]
analog-search decisions export
analog-search search QUERY
analog-search serve [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from analog_search.app.bootstrap import knowledge_service, models, search_service, services
from analog_search.app.config import Settings
from analog_search.domain.sources import SourceKind
from analog_search.infrastructure.sources import ksr
from analog_search.infrastructure.sources.store import Downloader
from analog_search.repositories.database import Database
from analog_search.repositories.sources import SourceRepository
from analog_search.services.sources import SourceService

MANUAL_URLS = {
    SourceKind.KSR: ksr.MANUAL_URL,
    SourceKind.KTRU: "https://int44.zakupki.gov.ru/eis-integration/services/getDocsIP (nsiKTRUNew)",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analog-search")
    commands = parser.add_subparsers(dest="command", required=True)

    db = commands.add_parser("db").add_subparsers(dest="action", required=True)
    db.add_parser("init", help="create extensions, schema and tables")

    sources = commands.add_parser("sources").add_subparsers(dest="action", required=True)
    fetch = sources.add_parser("fetch", help="download the newest free release")
    fetch.add_argument("source", choices=[SourceKind.KSR, SourceKind.OKPD2, SourceKind.ETIM], type=SourceKind)
    register = sources.add_parser("register", help="add a file downloaded by hand as a release")
    register.add_argument("source", choices=[SourceKind.KSR, SourceKind.KTRU], type=SourceKind)
    register.add_argument("file", type=Path)
    register.add_argument("--version", required=True)
    register.add_argument("--published", type=date.fromisoformat)
    load = sources.add_parser("load", help="parse a release and save it to the DB")
    load.add_argument("source", choices=list(SourceKind), type=SourceKind)
    load.add_argument("--dir", type=Path, help="release directory (default: latest fetched)")
    load.add_argument("--force", action="store_true", help="reload even if the same file is loaded")
    sources.add_parser("status", help="loaded releases")

    knowledge = commands.add_parser("knowledge").add_subparsers(dest="action", required=True)
    knowledge.add_parser(
        "load", help="import research/*.yaml as a knowledge release and activate it, over the specialists' edits"
    )

    positions = commands.add_parser("positions").add_subparsers(dest="action", required=True)
    positions_load = positions.add_parser("load", help="parse the procurement history and save it")
    positions_load.add_argument("file", type=Path, nargs="?", help="workbook (default: data/positions.xlsx)")

    decisions = commands.add_parser("decisions").add_subparsers(dest="action", required=True)
    decisions_load = decisions.add_parser("load", help="a specialist's verdicts from a file into golden cards")
    decisions_load.add_argument("file", type=Path, nargs="?", help="default: demo/decisions.json")
    decisions_load.add_argument("--author", default="демо")
    decisions.add_parser("export", help="golden cards as evaluation queries (JSON)")

    search = commands.add_parser("search", help="one query, the API response as JSON")
    search.add_argument("query")
    serve = commands.add_parser("serve", help="the API and the page")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    settings = Settings()

    if args.command == "db":
        db = Database(settings.database_url)
        db.init_schema()
        db.close()
        print("schema ready")
        return

    if args.command == "serve":
        import uvicorn

        uvicorn.run("analog_search.app.api:create_app", factory=True, host=args.host, port=args.port)
        return

    if args.command == "search":
        from analog_search.app.api import to_response

        db = Database(settings.database_url)
        try:
            result = to_response(search_service(settings, db).search(args.query))
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        finally:
            db.close()
        return

    if args.command == "decisions":
        db = Database(settings.database_url)
        try:
            cards = services(settings, db).cards
            if args.action == "load":
                print(cards.load(args.file or settings.demo_decisions, args.author, settings.positions_file.name))
            else:
                print(json.dumps(cards.export(), ensure_ascii=False, indent=2))
        finally:
            db.close()
        return

    if args.command in ("knowledge", "positions"):
        db = Database(settings.database_url)
        try:
            # positions: unread hard attributes asked of the model, names embedded (ANALOG_LLM)
            service = knowledge_service(settings, db, models(settings) if args.command == "positions" else None)
            if args.command == "knowledge":
                knowledge, counts = service.load_knowledge()
                print(f"{knowledge.version}: {'already stored, activated' if counts is None else counts}")
            else:
                print(service.load_positions(args.file or settings.positions_file))
        finally:
            db.close()
        return

    if args.action in ("fetch", "register"):
        downloader = Downloader(settings.timeout_s, settings.request_delay_s)
        try:
            service = SourceService(settings.sources_dir, downloader=downloader)
            if args.action == "fetch":
                directory = service.fetch(args.source)
            else:
                directory = service.register(
                    args.source, args.file, args.version, args.published, MANUAL_URLS[args.source]
                )
        finally:
            downloader.close()
        print(directory)
        return

    db = Database(settings.database_url)
    try:
        repository = SourceRepository(db)
        if args.action == "status":
            for row in repository.releases():
                print(f"{row.source:6} {row.version:12} {row.published or ''!s:10} {row.counts}")
            return
        counts = SourceService(settings.sources_dir, store=repository).load(args.source, args.dir, args.force)
        print("already loaded" if counts is None else counts)
    finally:
        db.close()


if __name__ == "__main__":
    main()
