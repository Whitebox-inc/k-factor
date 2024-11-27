import matplotlib.pyplot as plt
import networkx as nx
import os


def create_kay_architecture_diagram():
    """Creates a visualization of the Kay agent architecture using matplotlib."""

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with their positions
    # Main components
    pos = {
        "user": (0.5, 1),
        "kay": (0.5, 0.8),
        "llm": (0.2, 0.6),
        "tools": (0.8, 0.6),
        "memory": (0.5, 0.4),
        # Tool Generation Pipeline
        "asi_repl": (0.8, 0.8),
        "api_docs": (1.0, 1.0),
        "tools_json": (1.0, 0.6),
        # Request Handling
        "execute": (0.8, 0.4),
        "api": (1.0, 0.4),
        "response": (0.8, 0.2),
    }

    # Add nodes
    nodes = {
        "user": "User Input",
        "kay": "Kay Agent\n(LangChain Agent)",
        "llm": "AI21 LLM\n(Language Model)",
        "tools": "Tool Registry",
        "memory": "Conversation Memory",
        "asi_repl": "ASI REPL\nGenerator",
        "api_docs": "API Documentation",
        "tools_json": "generated_tools.json",
        "execute": "Execute Request",
        "api": "External API",
        "response": "Process Response",
    }

    for node, label in nodes.items():
        G.add_node(node, label=label)

    # Add edges with labels
    edges = [
        ("user", "kay", "input"),
        ("kay", "llm", "prompt"),
        ("llm", "kay", "response"),
        ("kay", "tools", "use tools"),
        ("kay", "memory", "store/retrieve"),
        ("api_docs", "asi_repl", "parse"),
        ("asi_repl", "tools_json", "generate"),
        ("tools_json", "tools", "load"),
        ("tools", "execute", "execute tool"),
        ("execute", "api", "request"),
        ("api", "response", "data"),
        ("response", "kay", "tool result"),
    ]

    G.add_edges_from([(u, v) for u, v, _ in edges])

    # Create the plot
    plt.figure(figsize=(15, 10))

    # Draw the network
    nx.draw(
        G,
        pos,
        node_color="lightblue",
        node_size=3000,
        arrows=True,
        edge_color="gray",
        width=2,
        arrowsize=20,
    )

    # Add node labels
    nx.draw_networkx_labels(
        G, pos, {node: label for node, label in nodes.items()}, font_size=8
    )

    # Add edge labels
    edge_labels = {(u, v): label for u, v, label in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    # Add title and adjust layout
    plt.title("Kay Agent Architecture", pad=20, size=16)
    plt.axis("off")

    # Create subgraph boxes
    tool_gen_box = plt.Rectangle(
        (0.7, 0.7),
        0.4,
        0.4,
        fill=False,
        linestyle="--",
        color="gray",
        label="Tool Generation Pipeline",
    )
    request_box = plt.Rectangle(
        (0.7, 0.1),
        0.4,
        0.4,
        fill=False,
        linestyle="--",
        color="gray",
        label="Request Handling",
    )

    # Add boxes to plot
    plt.gca().add_patch(tool_gen_box)
    plt.gca().add_patch(request_box)

    # Add legend
    plt.legend(loc="upper left", bbox_to_anchor=(0.1, 0.1))

    # Save the diagram
    os.makedirs("docs", exist_ok=True)
    output_path = os.path.join("docs", "kay_architecture.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Architecture diagram saved to {output_path}")


if __name__ == "__main__":
    create_kay_architecture_diagram()
