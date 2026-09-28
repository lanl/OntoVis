from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from rdflib import Graph, Literal, Namespace, RDF, XSD

from ontology import Ontology

EX = Namespace("http://example.org/dataset-kg#")

RAW_FILENAME_PATTERN = re.compile(
    r"^(?P<name>.+)_(?P<x>\d+)x(?P<y>\d+)x(?P<z>\d+)_(?P<dtype>[a-zA-Z0-9]+)$"
)

DTYPE_ALIASES = {
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "uint32": np.uint32,
    "int32": np.int32,
    "float32": np.float32,
    "float64": np.float64,
}


@dataclass(frozen=True)
class RawVolume:
    name: str
    size_x: int
    size_y: int
    size_z: int
    dtype: np.dtype
    data: np.ndarray

    @property
    def minimum(self) -> float:
        return float(self.data.min())

    @property
    def maximum(self) -> float:
        return float(self.data.max())


@dataclass(frozen=True)
class FieldMetadata:
    name: str
    units: str | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    description: str | None = None
    capture_method: str | None = None
    spacing: tuple[float, float, float] | None = None
    field_name: str | None = None
    field_units: str | None = None
    additional_fields: list[FieldMetadata] = field(default_factory=list)


def parse_raw_volume(path: Path) -> RawVolume:
    match = RAW_FILENAME_PATTERN.match(path.stem)

    if not match:
        raise ValueError(
            f"Filename {path.name!r} does not match the expected "
            "'<name>_<x>x<y>x<z>_<dtype>.raw' convention."
        )

    name = match.group("name")
    size_x = int(match.group("x"))
    size_y = int(match.group("y"))
    size_z = int(match.group("z"))
    dtype_name = match.group("dtype")

    if dtype_name not in DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported dtype {dtype_name!r} in filename {path.name!r}. "
            f"Supported dtypes: {', '.join(sorted(DTYPE_ALIASES))}"
        )

    dtype = np.dtype(DTYPE_ALIASES[dtype_name])
    expected_bytes = size_x * size_y * size_z * dtype.itemsize
    actual_bytes = path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch for {path.name!r}: expected {expected_bytes} bytes "
            f"for {size_x}x{size_y}x{size_z} {dtype_name} data, found {actual_bytes} bytes."
        )

    data = np.fromfile(path, dtype=dtype)

    return RawVolume(
        name=name,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        dtype=dtype,
        data=data,
    )


def _parse_field_metadata(entry: dict, *, context: str) -> FieldMetadata:
    if "name" not in entry:
        raise ValueError(f"{context} is missing required key 'name'.")

    return FieldMetadata(
        name=entry["name"],
        units=entry.get("units"),
        minimum=entry.get("minimum"),
        maximum=entry.get("maximum"),
    )


def load_metadata(dataset_path: Path) -> DatasetMetadata | None:
    sidecar_path = dataset_path.with_name(dataset_path.name + ".meta.yaml")

    if not sidecar_path.exists():
        return None

    data = yaml.safe_load(sidecar_path.read_text())

    if not isinstance(data, dict):
        raise ValueError(f"Sidecar {sidecar_path.name!r} must contain a YAML mapping.")

    spacing = data.get("spacing")
    if spacing is not None:
        if not (isinstance(spacing, (list, tuple)) and len(spacing) == 3):
            raise ValueError(
                f"Sidecar {sidecar_path.name!r}: 'spacing' must be a 3-element list [x, y, z]."
            )
        spacing = tuple(float(v) for v in spacing)

    field_entry = data.get("field")
    field_name = None
    field_units = None
    if field_entry is not None:
        if not isinstance(field_entry, dict):
            raise ValueError(f"Sidecar {sidecar_path.name!r}: 'field' must be a mapping.")
        field_name = field_entry.get("name")
        field_units = field_entry.get("units")

    additional_fields_data = data.get("additional_fields", [])
    if not isinstance(additional_fields_data, list):
        raise ValueError(f"Sidecar {sidecar_path.name!r}: 'additional_fields' must be a list.")

    additional_fields = [
        _parse_field_metadata(entry, context=f"Sidecar {sidecar_path.name!r} additional_fields[{i}]")
        for i, entry in enumerate(additional_fields_data)
    ]

    return DatasetMetadata(
        description=data.get("description"),
        capture_method=data.get("capture_method"),
        spacing=spacing,
        field_name=field_name,
        field_units=field_units,
        additional_fields=additional_fields,
    )


def validate_cardinalities(ontology: Ontology, field_count: int) -> None:
    has_rules = [rule for rule in ontology.rules_for("Dataset") if rule.relation == "has"]
    grid_rule = next((r for r in has_rules if r.target == "Grid"), None)
    field_rule = next((r for r in has_rules if r.target == "Field"), None)

    if grid_rule is None:
        raise ValueError("Ontology has no 'Dataset <has> Grid' rule.")
    if field_rule is None:
        raise ValueError("Ontology has no 'Dataset <has> Field' rule.")

    grid_count = 1

    if not (grid_rule.cardinality.minimum <= grid_count):
        raise ValueError("Dataset must have at least one Grid per the ontology.")
    if grid_rule.cardinality.maximum is not None and grid_count > grid_rule.cardinality.maximum:
        raise ValueError("Dataset has too many Grids per the ontology.")

    if not (field_rule.cardinality.minimum <= field_count):
        raise ValueError("Dataset must have at least one Field per the ontology.")
    if field_rule.cardinality.maximum is not None and field_count > field_rule.cardinality.maximum:
        raise ValueError("Dataset has too many Fields per the ontology.")


def _add_field(
    graph: Graph,
    dataset_uri,
    field_uri,
    *,
    name: str,
    minimum: float | None,
    maximum: float | None,
    units: str | None,
) -> None:
    graph.add((dataset_uri, EX.has, field_uri))
    graph.add((field_uri, RDF.type, EX.Field))
    graph.add((field_uri, RDF.type, EX.Scalar))
    graph.add((field_uri, EX.data_type, EX.Scalar))
    graph.add((field_uri, EX.association, EX.Point))
    graph.add((field_uri, EX.name, Literal(name)))
    graph.add((field_uri, EX.components, Literal(1, datatype=XSD.integer)))

    if minimum is not None:
        graph.add((field_uri, EX.minimum, Literal(minimum, datatype=XSD.double)))
    if maximum is not None:
        graph.add((field_uri, EX.maximum, Literal(maximum, datatype=XSD.double)))
    if units is not None:
        graph.add((field_uri, EX.units, Literal(units)))


def build_graph(
    volume: RawVolume,
    description: str,
    field_name: str,
    metadata: DatasetMetadata | None = None,
) -> Graph:
    graph = Graph()
    graph.bind("ex", EX)

    dataset_uri = EX[f"dataset_{volume.name}"]
    grid_uri = EX[f"grid_{volume.name}"]
    field_uri = EX[f"field_{volume.name}_{field_name}"]

    graph.add((dataset_uri, RDF.type, EX.Dataset))
    graph.add((dataset_uri, EX.description, Literal(description)))
    graph.add((dataset_uri, EX.temporal_type, EX.Static))
    graph.add((dataset_uri, EX.has, grid_uri))

    graph.add((grid_uri, RDF.type, EX.Grid))
    graph.add((grid_uri, RDF.type, EX.UniformGrid))
    graph.add((grid_uri, EX.grid_type, EX.UniformGrid))
    graph.add((grid_uri, EX.spatial_dimensions, Literal(3, datatype=XSD.integer)))
    graph.add((grid_uri, EX.size_x, Literal(volume.size_x, datatype=XSD.integer)))
    graph.add((grid_uri, EX.size_y, Literal(volume.size_y, datatype=XSD.integer)))
    graph.add((grid_uri, EX.size_z, Literal(volume.size_z, datatype=XSD.integer)))

    field_units = metadata.field_units if metadata else None

    _add_field(
        graph, dataset_uri, field_uri,
        name=field_name, minimum=volume.minimum, maximum=volume.maximum, units=field_units,
    )

    if metadata:
        if metadata.capture_method is not None:
            graph.add((dataset_uri, EX.capture_method, Literal(metadata.capture_method)))

        if metadata.spacing is not None:
            spacing_x, spacing_y, spacing_z = metadata.spacing
            graph.add((grid_uri, EX.spacing_x, Literal(spacing_x, datatype=XSD.double)))
            graph.add((grid_uri, EX.spacing_y, Literal(spacing_y, datatype=XSD.double)))
            graph.add((grid_uri, EX.spacing_z, Literal(spacing_z, datatype=XSD.double)))

        for extra in metadata.additional_fields:
            extra_uri = EX[f"field_{volume.name}_{extra.name}"]
            _add_field(
                graph, dataset_uri, extra_uri,
                name=extra.name, minimum=extra.minimum, maximum=extra.maximum, units=extra.units,
            )

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct a knowledge graph from a raw volume dataset, "
        "conforming to dataset_ontology.md."
    )
    parser.add_argument("dataset", type=Path, help="Path to a .raw volume file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .ttl path (default: <dataset_stem>.ttl). Use '-' for stdout.",
    )
    parser.add_argument(
        "--description", default=None,
        help="Human-readable description of the dataset. Overrides the sidecar's description.",
    )
    parser.add_argument(
        "--field-name", default=None,
        help="Name for the dataset's scalar field (default: sidecar value, or 'intensity').",
    )
    parser.add_argument(
        "--ontology", type=Path, default=Path(__file__).with_name("dataset_ontology.md"),
        help="Path to the ontology file used for validation.",
    )
    args = parser.parse_args()

    ontology = Ontology.from_file(args.ontology)

    volume = parse_raw_volume(args.dataset)
    metadata = load_metadata(args.dataset)

    description = (
        args.description
        or (metadata.description if metadata else None)
        or f"Raw volume dataset '{volume.name}'."
    )
    field_name = (
        args.field_name
        or (metadata.field_name if metadata else None)
        or "intensity"
    )

    field_count = 1 + (len(metadata.additional_fields) if metadata else 0)
    validate_cardinalities(ontology, field_count=field_count)

    graph = build_graph(volume, description=description, field_name=field_name, metadata=metadata)
    turtle = graph.serialize(format="turtle")

    if args.output is None:
        output_path = args.dataset.with_suffix(".ttl")
    elif str(args.output) == "-":
        output_path = None
    else:
        output_path = args.output

    if output_path is None:
        sys.stdout.write(turtle)
    else:
        output_path.write_text(turtle)
        print(f"Wrote knowledge graph to {output_path}")


if __name__ == "__main__":
    main()
