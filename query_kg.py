from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdflib import Graph, URIRef

DEFAULT_PREFIXES = """
PREFIX ex: <http://example.org/dataset-kg#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""


def _format_term(graph: Graph, term) -> str:
    if term is None:
        return ""
    if isinstance(term, URIRef):
        try:
            return graph.namespace_manager.normalizeUri(term)
        except Exception:
            return str(term)
    return str(term)


def run_query(graph: Graph, query: str) -> None:
    results = graph.query(DEFAULT_PREFIXES + query)

    if results.vars:
        header = [str(v) for v in results.vars]
        rows = [[_format_term(graph, row[v]) for v in results.vars] for row in results]
        widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]

        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        print("  ".join("-" * w for w in widths))
        for row in rows:
            print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

        print(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''})")
    elif results.type == "ASK":
        print(results.askAnswer)
    else:
        print("OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a SPARQL query against a knowledge graph (.ttl) produced by kg_builder.py. "
        "The prefix 'ex:' for http://example.org/dataset-kg# is predefined."
    )
    parser.add_argument("ttl", type=Path, help="Path to a .ttl knowledge graph file")
    parser.add_argument(
        "--sparql", default=None,
        help="SPARQL query to run. If omitted, reads the query from stdin.",
    )
    args = parser.parse_args()

    query = args.sparql or sys.stdin.read()

    if not query.strip():
        parser.error("No SPARQL query provided (use --sparql or pipe a query via stdin).")

    graph = Graph()
    graph.parse(args.ttl, format="turtle")

    run_query(graph, query)


if __name__ == "__main__":
    main()
