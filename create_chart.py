# complex_app_graph.py

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def create_complex_graph():
    # Create a directed graph
    G = nx.DiGraph()

    # Define node categories
    categories = {
        "module": {"color": "#1f78b4", "shape": "s"},
        "class": {"color": "#33a02c", "shape": "s"},
        "function": {"color": "#e31a1c", "shape": "o"},
        "variable": {"color": "#ff7f00", "shape": "d"},
        "external": {"color": "#6a3d9a", "shape": "h"},
    }

    # Define Modules
    modules = ["asi_repl.py", "tool_module.py", "kay_agent.py"]

    # Define Classes within Modules
    classes = {
        "asi_repl.py": [
            "LanguageModel",
            "GenerateRequestNode",
            "ExecuteRequestNode",
            "CollectResponseNode",
            "PlanToolsNode",
            "WriteToolsNode",
            "GenerateDocsNode",
        ],
        "tool_module.py": ["ToolLoader", "ToolCreator"],
        "kay_agent.py": ["KayAgent"],
    }

    # Define Functions within Classes
    functions = {
        "LanguageModel": ["__init__", "generate_completion"],
        "GenerateRequestNode": ["__init__", "run"],
        "ExecuteRequestNode": ["run"],
        "CollectResponseNode": ["run"],
        "PlanToolsNode": ["run"],
        "WriteToolsNode": ["run"],
        "GenerateDocsNode": ["run"],
        "ToolLoader": ["load_tools"],
        "ToolCreator": ["create_tool"],
        "KayAgent": [
            "__init__",
            "handle_user_input",
            "process_response",
            "display_actions",
        ],
    }

    # Define Variables (simplified for demonstration)
    variables = [
        "api_docs",
        "target_goal",
        "endpoints",
        "model",
        "tools",
        "documentation",
        "generated_info",
        "response",
        "chat_history",
        "user_input",
        "visualized_response",
    ]

    # Define External Components
    external = ["User", "Output", "External API"]

    # Add Module Nodes
    for module in modules:
        G.add_node(module, category="module", label=module)

    # Add Class Nodes
    for module, class_list in classes.items():
        for cls in class_list:
            G.add_node(cls, category="class", label=cls)
            G.add_edge(module, cls, label="contains")

    # Add Function Nodes
    for cls, func_list in functions.items():
        for func in func_list:
            func_node = f"{cls}.{func}"
            G.add_node(func_node, category="function", label=func)
            G.add_edge(cls, func_node, label="has method")

    # Add Variable Nodes
    for var in variables:
        G.add_node(var, category="variable", label=var)

    # Add External Component Nodes
    for ext in external:
        G.add_node(ext, category="external", label=ext)

    # Define Edges between Functions and Variables (Data Flow)
    # This is a simplified representation
    data_flow = [
        ("asi_repl.py", "LanguageModel", "uses"),
        ("LanguageModel.__init__", "api_key", "initializes"),
        ("LanguageModel.generate_completion", "prompt", "uses"),
        ("GenerateRequestNode.run", "endpoint", "uses"),
        ("GenerateRequestNode.run", "api_docs", "uses"),
        ("GenerateRequestNode.run", "generated_info", "produces"),
        ("ExecuteRequestNode.run", "method", "uses"),
        ("ExecuteRequestNode.run", "headers", "uses"),
        ("ExecuteRequestNode.run", "params", "uses"),
        ("ExecuteRequestNode.run", "data", "uses"),
        ("ExecuteRequestNode.run", "response", "produces"),
        ("CollectResponseNode.run", "response", "uses"),
        ("CollectResponseNode.run", "target_goal", "uses"),
        ("CollectResponseNode.run", "data", "produces"),
        ("PlanToolsNode.run", "response", "uses"),
        ("PlanToolsNode.run", "tools", "produces"),
        ("WriteToolsNode.run", "tools", "uses"),
        ("WriteToolsNode.run", "generated_tools.json", "writes to"),
        ("GenerateDocsNode.run", "tools", "uses"),
        ("GenerateDocsNode.run", "documentation", "produces"),
        ("ToolLoader.load_tools", "generated_tools.json", "reads from"),
        ("ToolLoader.load_tools", "tools", "produces"),
        ("ToolCreator.create_tool", "tool_def", "uses"),
        ("ToolCreator.create_tool", "tool_instance", "produces"),
        ("KayAgent.__init__", "llm", "uses"),
        ("KayAgent.__init__", "tools", "uses"),
        ("KayAgent.handle_user_input", "user_input", "uses"),
        ("KayAgent.handle_user_input", "response", "produces"),
        ("KayAgent.process_response", "response", "uses"),
        ("KayAgent.process_response", "visualized_response", "produces"),
        ("KayAgent.display_actions", "chat_history", "uses"),
        ("KayAgent", "AgentExecutor", "interacts with"),
        ("AgentExecutor", "AI21 LLM", "uses"),
        ("AgentExecutor", "ConversationBufferMemory", "uses"),
        ("AgentExecutor", "KayAgent", "runs"),
        ("User", "KayAgent.handle_user_input", "sends input to"),
        ("KayAgent.handle_user_input", "Output", "sends output to"),
        ("ExecuteRequestNode.run", "External API", "calls"),
        ("External API", "ExecuteRequestNode.run", "returns response to"),
    ]

    for src, dst, label in data_flow:
        G.add_edge(src, dst, label=label)

    # Define positions using spring layout
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    # Prepare node colors and shapes
    color_map = []
    shape_map = {}
    for node, data in G.nodes(data=True):
        category = data.get(
            "category", "external"
        )  # Default to 'external' if category is not set
        color_map.append(categories[category]["color"])
        shape_map[node] = categories[category]["shape"]

    # Since NetworkX doesn't support multiple shapes in a single draw call,
    # we'll draw nodes of each shape separately
    plt.figure(figsize=(30, 20))
    plt.axis("off")
    plt.title("Complex App Architecture", color="white", fontsize=20, pad=20)

    # Set background color
    ax = plt.gca()
    ax.set_facecolor("black")
    fig = plt.gcf()
    fig.patch.set_facecolor("black")

    # Draw nodes by category (shape)
    for category, attrs in categories.items():
        nodes = [
            n
            for n, d in G.nodes(data=True)
            if d.get("category", "external") == category
        ]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=[categories[category]["color"]] * len(nodes),
            node_shape=attrs["shape"],
            node_size=2000,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
        )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="white",
        arrows=True,
        arrowstyle="->",
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw labels
    node_labels = {}
    for node, data in G.nodes(data=True):
        if "label" in data:
            node_labels[node] = data["label"]
        else:
            # If there's no label, use the node name itself
            node_labels[node] = node

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=10,
        font_color="white",
        font_family="sans-serif",
    )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="white", font_size=8, label_pos=0.5
    )

    plt.tight_layout()
    try:
        plt.savefig(
            "complex_app_graph_networkx.png", format="png", dpi=300, bbox_inches="tight"
        )
        print("Graph generated successfully as 'complex_app_graph_networkx.png'.")
    except Exception as e:
        print(f"An error occurred while generating the graph: {e}")
    plt.show()


if __name__ == "__main__":
    create_complex_graph()
