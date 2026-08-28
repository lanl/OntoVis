import matplotlib.pyplot as plt
import networkx as nx


class GraphVisualizer:
    """Handles visualization of knowledge graphs."""

    @staticmethod
    def visualize(graph, output_path="knowledge_graph.png", figsize=(12, 8)):
        """Visualize a knowledge graph and save to file.

        Args:
            graph: NetworkX DiGraph object
            output_path: Path to save the visualization
            figsize: Figure size tuple (width, height)
        """
        plt.figure(figsize=figsize)

        # Create layout
        pos = nx.spring_layout(graph, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            node_color='lightblue',
            node_size=3000,
            alpha=0.9
        )

        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            width=2
        )

        # Draw node labels
        nx.draw_networkx_labels(
            graph, pos,
            font_size=10,
            font_weight='bold'
        )

        # Draw edge labels (relations)
        edge_labels = nx.get_edge_attributes(graph, 'relation')
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels,
            font_size=8,
            font_color='red'
        )

        plt.title("Knowledge Graph", size=16, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph visualization saved to: {output_path}")
