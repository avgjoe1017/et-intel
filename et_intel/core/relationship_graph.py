"""
Relationship Graph Visualization
Uses NetworkX to create visual relationship maps between entities
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pathlib import Path
import json
from .. import config

class RelationshipGraph:
    """
    Creates network graphs of entity relationships
    """
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def build_graph_from_entities(self, entities: Dict, comments_df=None) -> nx.Graph:
        """
        Build NetworkX graph from extracted entities
        
        Args:
            entities: Dict from EntityExtractor with 'people', 'shows', 'couples'
            comments_df: Optional DataFrame to calculate edge weights from co-mentions
        
        Returns:
            NetworkX Graph object
        """
        self.graph = nx.Graph()
        
        # Add nodes (people and shows)
        for person, count, _, _ in entities.get('people', []):
            self.graph.add_node(person, type='person', mentions=count)
        
        for show, count, _, _ in entities.get('shows', []):
            self.graph.add_node(show, type='show', mentions=count)
        
        # Add edges from couples/relationships
        for couple in entities.get('couples', []):
            if len(couple) >= 3:
                person1, person2, co_count = couple[0], couple[1], couple[2]
                if person1 in self.graph and person2 in self.graph:
                    self.graph.add_edge(person1, person2, weight=co_count, type='relationship')
        
        # Calculate co-mention weights from comments if provided
        if comments_df is not None and 'comment_text' in comments_df.columns:
            self._add_co_mention_edges(comments_df, entities)
        
        return self.graph
    
    def _add_co_mention_edges(self, df, entities: Dict):
        """Add edges based on co-mentions in comments"""
        people = [p[0] for p in entities.get('people', [])]
        
        # Count co-mentions
        co_mentions = {}
        for _, row in df.iterrows():
            comment = str(row['comment_text']).lower()
            mentioned = [p for p in people if p.lower() in comment]
            
            # Create edges for all pairs of mentioned entities
            for i, person1 in enumerate(mentioned):
                for person2 in mentioned[i+1:]:
                    pair = tuple(sorted([person1, person2]))
                    co_mentions[pair] = co_mentions.get(pair, 0) + 1
        
        # Add edges with weights
        for (person1, person2), count in co_mentions.items():
            if person1 in self.graph and person2 in self.graph:
                if self.graph.has_edge(person1, person2):
                    # Update weight
                    current_weight = self.graph[person1][person2].get('weight', 0)
                    self.graph[person1][person2]['weight'] = current_weight + count
                else:
                    self.graph.add_edge(person1, person2, weight=count, type='co_mention')
    
    def visualize(self, output_path: Path = None, layout='spring', figsize=(12, 8)) -> Path:
        """
        Create visualization of the relationship graph
        
        Args:
            output_path: Where to save the image
            layout: Graph layout ('spring', 'circular', 'kamada_kawai')
            figsize: Figure size tuple
        
        Returns:
            Path to saved image
        """
        if len(self.graph) == 0:
            raise ValueError("Graph is empty. Build graph first with build_graph_from_entities()")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Separate nodes by type
        person_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'person']
        show_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'show']
        
        # Draw edges
        edges = self.graph.edges()
        edge_weights = [self.graph[u][v].get('weight', 1) for u, v in edges]
        
        nx.draw_networkx_edges(
            self.graph, pos,
            width=[w/10 for w in edge_weights],  # Scale edge widths
            alpha=0.3,
            edge_color='gray',
            ax=ax
        )
        
        # Draw nodes
        if person_nodes:
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=person_nodes,
                node_color='#3498db',
                node_size=1000,
                alpha=0.8,
                ax=ax
            )
        
        if show_nodes:
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=show_nodes,
                node_color='#e74c3c',
                node_size=1000,
                alpha=0.8,
                ax=ax
            )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title("Entity Relationship Network", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Save
        if output_path is None:
            output_path = config.REPORTS_DIR / "charts" / "relationship_graph.png"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def get_centrality_metrics(self) -> Dict:
        """
        Calculate centrality metrics to find most influential entities
        
        Returns:
            Dict with degree, betweenness, and closeness centrality
        """
        if len(self.graph) == 0:
            return {}
        
        degree = nx.degree_centrality(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)
        closeness = nx.closeness_centrality(self.graph)
        
        return {
            'degree': degree,
            'betweenness': betweenness,
            'closeness': closeness
        }
    
    def get_communities(self) -> List[List[str]]:
        """
        Detect communities (clusters of related entities)
        
        Returns:
            List of communities, each community is a list of node names
        """
        if len(self.graph) == 0:
            return []
        
        try:
            communities = nx.community.greedy_modularity_communities(self.graph)
            return [list(community) for community in communities]
        except Exception:
            # Fallback if algorithm fails
            return [[node] for node in self.graph.nodes()]

