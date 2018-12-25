import csv
import networkx as nx


# read csv file
def read_file(filename):
    csvFile = open(filename, "r")
    reader = csv.reader(csvFile)
    result = {}
    for item in reader:
        if 'edges' in filename:
            if int(item[0]) not in result.keys():
                result[int(item[0])] = []
                result[int(item[0])].append(int(item[1]))
            else:
                result[int(item[0])].append(int(item[1]))
        else:
            result[int(item[0])] = int(item[0])
    csvFile.close()
    return result


def main():
    edges_dict = read_file("BlogCatalog-dataset/data/edges.csv")  # edge_dictionary {start: [end]}
    node_label_dict = read_file("BlogCatalog-dataset/data/group-edges.csv")  # node_label {node: label}
    node_dict = read_file("BlogCatalog-dataset/data/nodes.csv")  # node_dictionary {node: node}
    group_dict = read_file("BlogCatalog-dataset/data/groups.csv")  # group_dictionary {group: group}
    # generate whole graph
    G = nx.Graph(label=len(group_dict.keys()))
    for node in node_dict.keys():
        label = node_label_dict[node]
        G.add_node(node, label=label)
    for node in edges_dict.keys():
        ends = edges_dict[node]
        start = node
        for end in ends:
            G.add_edge(start, end)
    print(G.graph)
    nx.write_gexf(G, path="../data/data.gexf")


if __name__ == '__main__':
    main()
