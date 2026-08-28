import networkx as nx


class KnowledgeGraph:
    """A simple knowledge graph implementation using NetworkX."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity, entity_type=None):
        """Add a node (entity) to the graph.

        Args:
            entity: The entity name/identifier
            entity_type: Optional type classification for the entity
        """
        self.graph.add_node(entity, type=entity_type)

    def add_relation(self, source, relation, target):
        """Add an edge (relationship) between two entities.

        Args:
            source: Source entity
            relation: The relationship/predicate
            target: Target entity
        """
        if source not in self.graph:
            self.add_entity(source)
        if target not in self.graph:
            self.add_entity(target)
        self.graph.add_edge(source, target, relation=relation)

    def get_neighbors(self, entity):
        """Get all entities connected to a given entity.

        Args:
            entity: The entity to query

        Returns:
            List of neighboring entities
        """
        if entity in self.graph:
            return list(self.graph.neighbors(entity))
        return []

    def get_relations(self, source, target):
        """Get the relation between two entities.

        Args:
            source: Source entity
            target: Target entity

        Returns:
            The relation label if exists, None otherwise
        """
        if self.graph.has_edge(source, target):
            return self.graph[source][target]['relation']
        return None

    def get_all_entities(self):
        """Get all entities in the graph.

        Returns:
            List of all entity nodes
        """
        return list(self.graph.nodes())

    def get_all_relations(self):
        """Get all relationships in the graph.

        Returns:
            List of tuples (source, target, relation)
        """
        relations = []
        for source, target, data in self.graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            relations.append((source, target, relation))
        return relations

    def print_graph(self):
        """Print all entities and relationships in a readable format."""
        print("=== Knowledge Graph ===")
        print(f"Entities: {self.get_all_entities()}")
        print(f"\nRelationships:")
        for source, target, relation in self.get_all_relations():
            print(f"  {source} --[{relation}]--> {target}")
