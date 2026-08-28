from ontovis import KnowledgeGraph
from ontovis.visualizer import GraphVisualizer


def main():
    # Create a simple example knowledge graph
    kg = KnowledgeGraph()

    # Add some example data
    kg.add_relation("Python", "is_a", "Programming Language")
    kg.add_relation("Python", "created_by", "Guido van Rossum")
    kg.add_relation("NetworkX", "is_a", "Library")
    kg.add_relation("NetworkX", "written_in", "Python")
    kg.add_relation("Knowledge Graph", "implemented_with", "NetworkX")
    kg.add_relation("Knowledge Graph", "stores", "Entities")
    kg.add_relation("Knowledge Graph", "stores", "Relations")

    # Print the graph
    kg.print_graph()

    # Visualize
    print("\nGenerating visualization...")
    GraphVisualizer.visualize(kg.graph, output_path="knowledge_graph.png")


if __name__ == "__main__":
    main()
