import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph():
    G = nx.Graph()
    
    # Example knowledge graph for stroke prevention recommendations
    G.add_edge('High Risk', 'Exercise Regularly', recommendation='Engage in moderate exercise 30 mins daily')
    G.add_edge('High Risk', 'Diet Control', recommendation='Adopt a low-fat, low-sodium diet')
    G.add_edge('Low Risk', 'Healthy Lifestyle', recommendation='Maintain a balanced diet and stay active')
    
    # Draw the graph
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'recommendation')
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Knowledge Graph: Stroke Prevention Recommendations")
    plt.show()

# Call the visualization function
visualize_graph()
