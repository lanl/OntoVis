from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdflib import Graph, Literal, RDF, URIRef

EX_PREFIX = "http://example.org/dataset-kg#"

NODE_TYPE_COLORS = {
    "Dataset": "series-1",
    "Grid": "series-2",
    "Field": "series-3",
}


def local_name(term) -> str:
    text = str(term)
    return text[len(EX_PREFIX):] if text.startswith(EX_PREFIX) else text


def primary_type(graph: Graph, subject: URIRef) -> str:
    for _, _, obj in graph.triples((subject, RDF.type, None)):
        name = local_name(obj)
        if name in NODE_TYPE_COLORS:
            return name
    return "Other"


def build_graph_data(graph: Graph) -> dict:
    subjects = sorted({s for s in graph.subjects() if isinstance(s, URIRef)}, key=str)

    nodes = []
    node_index = {}
    for subject in subjects:
        node_type = primary_type(graph, subject)
        properties = []
        for _, predicate, obj in graph.triples((subject, None, None)):
            pred_name = local_name(predicate)
            if pred_name in ("type", "has"):
                continue
            if isinstance(obj, Literal):
                properties.append((pred_name, str(obj)))
            elif isinstance(obj, URIRef) and local_name(obj) not in {local_name(s) for s in subjects}:
                properties.append((pred_name, local_name(obj)))

        node_index[subject] = len(nodes)
        nodes.append({
            "id": local_name(subject),
            "type": node_type,
            "colorRole": NODE_TYPE_COLORS.get(node_type, "series-4"),
            "properties": properties,
        })

    links = []
    for subject, predicate, obj in graph.triples((None, None, None)):
        pred_name = local_name(predicate)
        if pred_name == "type":
            continue
        if isinstance(obj, URIRef) and obj in node_index and subject in node_index:
            links.append({
                "source": node_index[subject],
                "target": node_index[obj],
                "label": pred_name,
            })

    return {"nodes": nodes, "links": links}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Knowledge Graph — {title}</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:     #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --border:         rgba(11,11,11,0.10);
    --series-1:       #2a78d6;
    --series-2:       #eb6834;
    --series-3:       #1baf7a;
    --series-4:       #eda100;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:     #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --border:         rgba(255,255,255,0.10);
      --series-1:       #3987e5;
      --series-2:       #d95926;
      --series-3:       #199e70;
      --series-4:       #c98500;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --border:         rgba(255,255,255,0.10);
    --series-1:       #3987e5;
    --series-2:       #d95926;
    --series-3:       #199e70;
    --series-4:       #c98500;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page-plane);
    color: var(--text-primary);
  }}
  .viz-root {{
    max-width: 1000px;
    margin: 24px auto;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px 24px;
  }}
  h1 {{
    font-size: 16px;
    font-weight: 600;
    margin: 0 0 4px;
    color: var(--text-primary);
  }}
  .subtitle {{
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0 0 16px;
  }}
  .legend {{
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
    font-size: 12px;
    color: var(--text-secondary);
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }}
  svg {{ display: block; width: 100%; height: 480px; }}
  .link {{
    stroke: var(--gridline);
    stroke-width: 1.5px;
  }}
  .link-label {{
    fill: var(--text-muted);
    font-size: 10px;
  }}
  .node circle {{
    stroke: var(--surface-1);
    stroke-width: 2px;
    cursor: pointer;
  }}
  .node text {{
    fill: var(--text-primary);
    font-size: 12px;
    font-weight: 600;
    pointer-events: none;
  }}
  .tooltip {{
    position: absolute;
    display: none;
    max-width: 260px;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 12px;
    color: var(--text-secondary);
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    pointer-events: none;
    z-index: 10;
  }}
  .tooltip strong {{
    color: var(--text-primary);
    font-size: 13px;
  }}
  .tooltip table {{ margin-top: 6px; border-collapse: collapse; }}
  .tooltip td {{ padding: 1px 6px 1px 0; vertical-align: top; font-variant-numeric: tabular-nums; }}
  .tooltip td.key {{ color: var(--text-muted); white-space: nowrap; }}
</style>
</head>
<body>
<div class="viz-root" style="position: relative;">
  <h1>Knowledge Graph — {title}</h1>
  <p class="subtitle">Instances and relations from {ttl_filename}, per dataset_ontology.md</p>
  <div class="legend" id="legend"></div>
  <svg id="chart"></svg>
  <div class="tooltip" id="tooltip"></div>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {data_json};

const root = document.querySelector('.viz-root');
const colorOf = (role) => getComputedStyle(root).getPropertyValue('--' + role).trim();

const typeOrder = [...new Set(data.nodes.map(n => n.type))];
const legend = document.getElementById('legend');
typeOrder.forEach(type => {{
  const node = data.nodes.find(n => n.type === type);
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `<span class="legend-dot" style="background:${{colorOf(node.colorRole)}}"></span>${{type}}`;
  legend.appendChild(item);
}});

const svg = d3.select('#chart');
const width = svg.node().clientWidth || 900;
const height = 480;
svg.attr('viewBox', `0 0 ${{width}} ${{height}}`);

const simulation = d3.forceSimulation(data.nodes)
  .force('link', d3.forceLink(data.links).id((d, i) => i).distance(160).strength(0.6))
  .force('charge', d3.forceManyBody().strength(-420))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collide', d3.forceCollide(48));

const link = svg.append('g')
  .selectAll('line')
  .data(data.links)
  .join('line')
  .attr('class', 'link');

const linkLabel = svg.append('g')
  .selectAll('text')
  .data(data.links)
  .join('text')
  .attr('class', 'link-label')
  .text(d => d.label);

const node = svg.append('g')
  .selectAll('g')
  .data(data.nodes)
  .join('g')
  .attr('class', 'node')
  .call(d3.drag()
    .on('start', (event, d) => {{ if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on('drag', (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
    .on('end', (event, d) => {{ if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

node.append('circle')
  .attr('r', 26)
  .attr('fill', d => colorOf(d.colorRole));

node.append('text')
  .attr('text-anchor', 'middle')
  .attr('dy', 42)
  .text(d => d.id);

const tooltip = document.getElementById('tooltip');

node.on('mouseenter', (event, d) => {{
  let rows = d.properties.map(([k, v]) => `<tr><td class="key">${{k}}</td><td>${{v}}</td></tr>`).join('');
  tooltip.innerHTML = `<strong>${{d.id}}</strong> <span style="color:var(--text-muted)">(${{d.type}})</span>` +
    (rows ? `<table>${{rows}}</table>` : '');
  tooltip.style.display = 'block';
}}).on('mousemove', (event) => {{
  const bounds = root.getBoundingClientRect();
  tooltip.style.left = (event.clientX - bounds.left + 16) + 'px';
  tooltip.style.top = (event.clientY - bounds.top + 8) + 'px';
}}).on('mouseleave', () => {{
  tooltip.style.display = 'none';
}});

simulation.on('tick', () => {{
  link
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x)
    .attr('y2', d => d.target.y);

  linkLabel
    .attr('x', d => (d.source.x + d.target.x) / 2)
    .attr('y', d => (d.source.y + d.target.y) / 2 - 4);

  node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
}});
</script>
</body>
</html>
"""


def build_html(graph_data: dict, title: str, ttl_filename: str) -> str:
    return HTML_TEMPLATE.format(
        title=title,
        ttl_filename=ttl_filename,
        data_json=json.dumps(graph_data),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Turtle knowledge graph (from kg_builder.py) as an interactive HTML visualization."
    )
    parser.add_argument("ttl", type=Path, help="Path to a .ttl knowledge graph file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .html path (default: <ttl_stem>.html)")
    args = parser.parse_args()

    graph = Graph()
    graph.parse(args.ttl, format="turtle")

    graph_data = build_graph_data(graph)
    title = args.ttl.stem
    html = build_html(graph_data, title=title, ttl_filename=args.ttl.name)

    output_path = args.output or args.ttl.with_suffix(".html")
    output_path.write_text(html)
    print(f"Wrote visualization to {output_path}")


if __name__ == "__main__":
    main()
