from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import yaml


PRIMITIVE_TYPES = {
    "string",
    "integer",
    "number",
    "boolean",
}


@dataclass(frozen=True)
class Concept:
    name: str
    definition: str


@dataclass(frozen=True)
class Cardinality:
    minimum: int
    maximum: Optional[int]

    @classmethod
    def parse(cls, value: str) -> "Cardinality":
        value = value.strip()

        if ".." not in value:
            n = int(value)
            return cls(n, n)

        minimum, maximum = value.split("..", maxsplit=1)

        return cls(
            minimum=int(minimum),
            maximum=None if maximum == "*" else int(maximum),
        )

    def __str__(self) -> str:
        if self.maximum == self.minimum:
            return str(self.minimum)

        maximum = "*" if self.maximum is None else str(self.maximum)
        return f"{self.minimum}..{maximum}"


@dataclass(frozen=True)
class Rule:
    source: str
    relation: str
    cardinality: Cardinality
    target_kind: str
    target: str | tuple[str, ...]


class Ontology:
    RULE_PATTERN = re.compile(
        r"""
        ^(?P<source>\w+)
        \s*
        <(?P<relation>[^>]+)>
        \s*
        \[(?P<cardinality>[^\]]+)\]
        \s*
        (?P<target>.+)
        $
        """,
        re.VERBOSE,
    )

    def __init__(
        self,
        concepts: dict[str, Concept],
        rules: list[Rule],
    ):
        self.concepts = concepts
        self.rules = rules
        self._validate()

    @classmethod
    def from_file(cls, filename: str | Path) -> "Ontology":
        path = Path(filename)
        text = path.read_text()

        definitions_text, rules_text = cls._split_file(text)
        concepts = cls._parse_concepts(definitions_text)
        rules = cls._parse_rules(rules_text)

        return cls(concepts=concepts, rules=rules)

    @staticmethod
    def _split_file(text: str) -> tuple[str, str]:
        lines = text.splitlines()

        if not lines or lines[0].strip() != "---":
            raise ValueError("Ontology must begin with '---'.")

        try:
            end = next(
                i
                for i, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            )
        except StopIteration:
            raise ValueError("Could not find closing '---' for definitions.")

        definitions = "\n".join(lines[1:end])
        rules = "\n".join(lines[end + 1 :])
        return definitions, rules

    @classmethod
    def _parse_concepts(cls, text: str) -> dict[str, Concept]:
        data = yaml.safe_load(text)

        if isinstance(data, dict) and "definitions" in data:
            return cls._parse_concepts_list(data["definitions"])

        if isinstance(data, dict):
            return cls._parse_concepts_mapping(data)

        raise ValueError("Concept definitions must be a mapping.")

    @staticmethod
    def _parse_concepts_mapping(data: dict) -> dict[str, Concept]:
        concepts = {}

        for name, attributes in data.items():
            if not isinstance(attributes, dict):
                raise ValueError(f"Concept {name!r} must have attributes.")

            definition = attributes.get("definition")

            if not definition:
                raise ValueError(f"Concept {name!r} has no definition.")

            concepts[name] = Concept(
                name=name,
                definition=definition.strip(),
            )

        return concepts

    @staticmethod
    def _parse_concepts_list(entries: list) -> dict[str, Concept]:
        concepts = {}

        for entry in entries:
            if isinstance(entry, dict) and len(entry) == 1:
                (name, definition), = entry.items()
            elif isinstance(entry, str) and ":" in entry:
                name, definition = entry.split(":", maxsplit=1)
            else:
                raise ValueError(f"Invalid concept definition entry: {entry!r}")

            name = name.strip()
            definition = str(definition).strip()

            if not definition:
                raise ValueError(f"Concept {name!r} has no definition.")

            concepts[name] = Concept(name=name, definition=definition)

        return concepts

    @classmethod
    def _parse_rules(cls, text: str) -> list[Rule]:
        statements = cls._collect_statements(text)
        return [cls._parse_rule(statement) for statement in statements]

    @staticmethod
    def _collect_statements(text: str) -> list[str]:
        statements = []
        buffer: list[str] = []
        brace_depth = 0

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            buffer.append(line)
            brace_depth += line.count("{")
            brace_depth -= line.count("}")

            if brace_depth == 0:
                statements.append(" ".join(buffer))
                buffer = []

        if buffer:
            raise ValueError("Unterminated rule block.")

        return statements

    @classmethod
    def _parse_rule(cls, statement: str) -> Rule:
        match = cls.RULE_PATTERN.match(statement)

        if not match:
            raise ValueError(f"Invalid rule:\n{statement}")

        source = match.group("source")
        relation = match.group("relation").strip()
        cardinality = Cardinality.parse(match.group("cardinality"))
        target_text = match.group("target").strip()

        if target_text.startswith("{") and target_text.endswith("}"):
            contents = target_text[1:-1]
            values = tuple(
                value.strip()
                for value in contents.split(",")
                if value.strip()
            )
            target_kind = "enum"
            target = values
        elif target_text in PRIMITIVE_TYPES:
            target_kind = "primitive"
            target = target_text
        else:
            target_kind = "concept"
            target = target_text

        return Rule(
            source=source,
            relation=relation,
            cardinality=cardinality,
            target_kind=target_kind,
            target=target,
        )

    def _validate(self) -> None:
        for rule in self.rules:
            if rule.source not in self.concepts:
                raise ValueError(f"Unknown source concept: {rule.source}")

            if rule.target_kind == "concept":
                assert isinstance(rule.target, str)
                if rule.target not in self.concepts:
                    raise ValueError(f"Unknown target concept: {rule.target}")

            if rule.target_kind == "enum":
                assert isinstance(rule.target, tuple)
                for value in rule.target:
                    if value not in self.concepts and not value.isdigit():
                        raise ValueError(
                            f"Unknown enum value {value!r} in rule "
                            f"{rule.source} <{rule.relation}>"
                        )

    def describe(self, concept: str) -> Concept:
        return self.concepts[concept]

    def rules_for(self, concept: str) -> list[Rule]:
        return [rule for rule in self.rules if rule.source == concept]

    def relations_for(self, concept: str, relation: str) -> list[Rule]:
        return [
            rule
            for rule in self.rules
            if rule.source == concept and rule.relation == relation
        ]

    def print_summary(self) -> None:
        print("Concepts")
        print("========")

        for concept in self.concepts.values():
            print(f"\n{concept.name}")
            print(f"  {concept.definition}")

        print("\n\nRules")
        print("=====")

        for rule in self.rules:
            if rule.target_kind == "enum":
                target = "{" + ", ".join(rule.target) + "}"
            else:
                target = rule.target

            print(
                f"{rule.source} "
                f"<{rule.relation}> "
                f"[{rule.cardinality}] "
                f"{target}"
            )


if __name__ == "__main__":
    ontology_path = Path(__file__).with_name("ontology.txt")
    ontology = Ontology.from_file(ontology_path)
    ontology.print_summary()
