from __future__ import annotations

import argparse
import json
from pathlib import Path

from ontology import Ontology, Rule

NODE_COLOR_CYCLE = ["series-1", "series-2", "series-3", "series-4"]


def build_graph_data(ontology: Ontology) -> dict:
    node_index: dict[str, int] = {}
    nodes = []

    def ensure_node(name: str) -> int:
        if name not in node_index:
            concept = ontology.concepts.get(name)
            node_index[name] = len(nodes)
            nodes.append({
                "id": name,
                "definition": concept.definition if concept else "",
                "colorRole": NODE_COLOR_CYCLE[len(node_index) % len(NODE_COLOR_CYCLE)],
            })
        return node_index[name]

    for concept in ontology.concepts:
        ensure_node(concept)

    links = []

    def add_link(source: str, relation: str, target: str, cardinality: str) -> None:
        links.append({
            "source": ensure_node(source),
            "target": ensure_node(target),
            "label": relation,
            "cardinality": cardinality,
        })

    rule: Rule
    for rule in ontology.rules:
        if rule.target_kind == "concept":
            add_link(rule.source, rule.relation, rule.target, str(rule.cardinality))
        elif rule.target_kind == "enum":
            for value in rule.target:
                if value in ontology.concepts:
                    add_link(rule.source, rule.relation, value, str(rule.cardinality))

    return {"nodes": nodes, "links": links}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Ontology — {title}</title>
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
    max-width: 1400px;
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
  .controls {{
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 12px;
    font-size: 12px;
    color: var(--text-secondary);
  }}
  .controls input {{
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--page-plane);
    color: var(--text-primary);
  }}
  svg {{ display: block; width: 100%; height: 760px; }}
  .link {{
    stroke: var(--gridline);
    stroke-width: 1.4px;
  }}
  .link-label {{
    fill: var(--text-muted);
    font-size: 9px;
  }}
  .node circle {{
    stroke: var(--surface-1);
    stroke-width: 2px;
    cursor: pointer;
  }}
  .node text {{
    fill: var(--text-primary);
    font-size: 11px;
    font-weight: 600;
    pointer-events: none;
  }}
  .node.dim circle {{ opacity: 0.15; }}
  .node.dim text {{ opacity: 0.15; }}
  .link.dim {{ opacity: 0.08; }}
  .link-label.dim {{ opacity: 0.08; }}
  .tooltip {{
    position: absolute;
    display: none;
    max-width: 320px;
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
</style>
</head>
<body>
<div class="viz-root" style="position: relative;">
  <h1>Ontology — {title}</h1>
  <p class="subtitle">Concepts and rules from {source_filename}</p>
  <div class="controls">
    <label>Search: <input id="search" type="text" placeholder="concept name..." /></label>
    <span id="count"></span>
  </div>
  <svg id="chart"></svg>
  <div class="tooltip" id="tooltip"></div>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {data_json};

const root = document.querySelector('.viz-root');
const colorOf = (role) => getComputedStyle(root).getPropertyValue('--' + role).trim();

document.getElementById('count').textContent =
  data.nodes.length + ' concepts, ' + data.links.length + ' relations';

const svg = d3.select('#chart');
const width = svg.node().clientWidth || 1300;
const height = 760;
svg.attr('viewBox', `0 0 ${{width}} ${{height}}`);

const simulation = d3.forceSimulation(data.nodes)
  .force('link', d3.forceLink(data.links).id((d, i) => i).distance(130).strength(0.5))
  .force('charge', d3.forceManyBody().strength(-320))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collide', d3.forceCollide(42));

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
  .text(d => d.label + ' [' + d.cardinality + ']');

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
  .attr('r', 22)
  .attr('fill', d => colorOf(d.colorRole));

node.append('text')
  .attr('text-anchor', 'middle')
  .attr('dy', 36)
  .text(d => d.id);

const tooltip = document.getElementById('tooltip');

node.on('mouseenter', (event, d) => {{
  tooltip.innerHTML = `<strong>${{d.id}}</strong><div style="margin-top:6px;">${{d.definition || ''}}</div>`;
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

const searchInput = document.getElementById('search');
searchInput.addEventListener('input', () => {{
  const q = searchInput.value.trim().toLowerCase();
  if (!q) {{
    node.classed('dim', false);
    link.classed('dim', false);
    linkLabel.classed('dim', false);
    return;
  }}
  const matched = new Set(data.nodes.filter(n => n.id.toLowerCase().includes(q)).map(n => n.id));
  node.classed('dim', d => !matched.has(d.id));
  link.classed('dim', d => !matched.has(d.source.id) && !matched.has(d.target.id));
  linkLabel.classed('dim', d => !matched.has(d.source.id) && !matched.has(d.target.id));
}});
</script>
</body>
</html>
"""


def build_html(graph_data: dict, title: str, source_filename: str) -> str:
    return HTML_TEMPLATE.format(
        title=title,
        source_filename=source_filename,
        data_json=json.dumps(graph_data),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a human-editable ontology markdown file (concepts + rules, "
        "per dataset_ontology.md's format) as an interactive HTML knowledge graph."
    )
    parser.add_argument("ontology", type=Path, help="Path to an ontology .md/.txt file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .html path (default: <ontology_stem>.html)",
    )
    args = parser.parse_args()

    ontology = Ontology.from_file(args.ontology)
    graph_data = build_graph_data(ontology)
    title = args.ontology.stem
    html = build_html(graph_data, title=title, source_filename=args.ontology.name)

    output_path = args.output or args.ontology.with_suffix(".html")
    output_path.write_text(html)
    print(f"Wrote visualization to {output_path}")


if __name__ == "__main__":
    main()
