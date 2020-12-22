#to use this library, you need to "brew install graphviz"
from os import remove
from graphviz import Digraph
from sigfig import round

def createGraph(name, activation_values, weights, biases, error_metric):
    dot = Digraph(comment=name)
    nodes = []
    node_count = 0
    for layer in activation_values:
        nodes.append([])
        for value in layer:
            node_name = "N" + str(node_count)
            nodes[-1].append(node_name)
            dot.node(node_name, str(round(value,5)))
            node_count += 1

    node_count = 1
    bias_nodes = []
    for bias in biases:
        node_name = "B" + str(node_count)
        bias_nodes.append(node_name)
        dot.node(node_name, str(round(bias,5)), color="blue")
        node_count += 1

    #unpack weights
    flat_weights = []
    for weight_layer in weights:
        for weight_set in weight_layer:
            for weight in weight_set:
                flat_weights.append(weight)

    edges = []
    bias_edges = []
    for node_layer_index in range(1, len(nodes)):
        for prev_node in nodes[node_layer_index - 1]:
            for node in nodes[node_layer_index]:
                a = str(prev_node)
                b = node
                edges.append([a,b])
        if node_layer_index < len(nodes):
            for node in nodes[node_layer_index]:
                try:
                    a = bias_nodes[node_layer_index-1]
                    b = node
                    bias_edges.append([a, b])
                except :
                    print("Output layer has no bias")
    for node_pair_index in range(0, len(edges)):
        dot.edge(edges[node_pair_index][0], edges[node_pair_index][1], label=str(round(flat_weights[node_pair_index], 5)))
    for node_pair_index in range(0, len(bias_edges)):
        dot.edge(bias_edges[node_pair_index][0], bias_edges[node_pair_index][1], color="blue")

    if error_metric[1] == None:
        metric = "N/A"
    else :
        metric = str(round(error_metric[1], 5))
    dot.attr(label=r'\n\n' + error_metric[0] + ": " + metric)

    dot.render(name, view=True)
    remove(name)
