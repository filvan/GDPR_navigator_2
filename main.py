import openpyxl
import pandas as pd
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt


class LegalText:
    def __init__(self, name, translated=False, references=None):
        self.name = name
        self.translated = translated
        if references is None:
            references = []
        self.references = references

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()


def setup_graph_matrix(index2article_with_references: dict, article2index: dict, option: int = 0) -> np.ndarray:
    if not 0 <= option <= 3:
        print("Invalid value of the option. Please choose an option value between 0, 1, 2 and 3")
        exit(1)
    graph_matrix = np.zeros((len(index2article_with_references), len(index2article_with_references)), dtype=int)
    for i in range(len(index2article_with_references)):
        # graph_matrix[0][i] = i
        # graph_matrix[i][0] = i
        references = index2article_with_references[i].references
        if references:
            for reference in references:
                regex1 = r"^Art.\ \d+(\(.\))*$"
                regex2 = r"Art\.\ \d+-\d+"
                match1 = re.search(pattern=regex1, string=reference)
                if match1:
                    match1 = match1.string
                    match1 = re.sub(r"[a-zA-Z]+\.\s", "", match1)
                    if option == 2 or option == 3:
                        k = article2index[match1]
                        name = index2article_with_references[k].name
                        while match1 in name:
                            graph_matrix[i][k] = 1
                            k += 1
                            if k == len(index2article_with_references):
                                break
                            name = index2article_with_references[k].name
                    else:
                        reference_index = article2index[match1]
                        graph_matrix[i][reference_index] = 1
                else:
                    match2 = re.search(pattern=regex2, string=reference)
                    if match2:
                        match2 = match2.string
                        match2 = re.sub(r"[^\d+\-\d+]", "", match2)
                        match2 = match2.split("-")
                        if option == 1 or option == 3:
                            increment = 0
                            interval_start = article2index[match2[0]]
                            interval_end = article2index[match2[1]]
                            k = interval_start
                            name = index2article_with_references[k].name
                            while match2[1] in name:
                                increment += 1
                                k += 1
                                if k == len(index2article_with_references):
                                    break
                                name = index2article_with_references[k].name
                        else:
                            increment = 1
                            interval_start = int(match2[0])
                            interval_end = int(match2[1])
                        for j in range(interval_start, interval_end + increment):
                            if option == 1 or option == 3:
                                reference_index = j
                            else:
                                reference_index = article2index[str(j)]
                            graph_matrix[i][reference_index] = 1
    print("Graph matrix has been set up.\n")
    return graph_matrix


def setup_links() -> list:
    wb = openpyxl.load_workbook('GDPR_map.xlsx')
    ws = wb['Cartel1']
    links = []
    for i in range(2, ws.max_row + 1):
        link = ws.cell(row=i, column=6).hyperlink
        if link is None:
            print("No link found.")
        links.append(link.target)
    return links


def setup_dictionaries(df, index2references, index2article_with_references, article2index, article2references):
    for i in range(len(df)):
        row = df.loc[i]
        article, paragraph, point, translated, references, link = row
        if len(references) == 0:
            index2references[i] = []
        else:
            index2references[i] = references.split(', ')
        if len(point) > 0:
            name = article + '(' + paragraph + ')' + '(' + point + ')'
        elif len(paragraph) > 0:
            name = article + '(' + paragraph + ')'
        else:
            name = article
        index2article_with_references[i] = LegalText(name, translated, index2references[i])
        article2index[name] = i
        article2references[name] = index2references[i]


num_cycles = 0


def DFS(current_node, visited_nodes, visited_nodes2, visited_nodes3, recursion_stack, recursion_stack2,
        recursion_stack3: list, graph_matrix, index2article_with_references):
    global num_cycles
    current_node_name = index2article_with_references[current_node]
    visited_nodes[current_node] = True
    visited_nodes2[current_node] = current_node_name
    visited_nodes3.append(current_node_name)
    recursion_stack[current_node] = True
    recursion_stack2[current_node] = current_node_name
    recursion_stack3.append(current_node_name)
    # print("Visiting Art.", current_node_name)

    for adjacent_node in range(len(index2article_with_references)):
        adjacent_node_name = index2article_with_references[adjacent_node]
        if graph_matrix[current_node][adjacent_node] == 1:
            # print("Art.", current_node_name, "-> Art.", adjacent_node_name)
            if not visited_nodes[adjacent_node]:
                # print("Art.", current_node_name, "-> Art.", adjacent_node_name)
                DFS(adjacent_node, visited_nodes, visited_nodes2, visited_nodes3, recursion_stack, recursion_stack2,
                    recursion_stack3, graph_matrix,
                    index2article_with_references)
            elif recursion_stack[adjacent_node]:
                if graph_matrix[adjacent_node][current_node] == 1:
                    print("Cycle detected:", current_node_name, "<->", adjacent_node_name)
                else:
                    print("Cycle detected: ", end="")
                    for node in range(len(recursion_stack3)):
                        print(recursion_stack3[node], "-> ", end="")
                    print(recursion_stack2[adjacent_node])
                num_cycles += 1

    recursion_stack[current_node] = False
    recursion_stack2[current_node] = ''
    recursion_stack3.remove(current_node_name)
    # print("Visit on Art.", current_node_name, "completed.")


def detect_cycle(graph_matrix, index2article_with_references):
    num_nodes = len(index2article_with_references)
    global num_cycles
    num_cycles = 0

    visited_nodes = [False] * num_nodes
    visited_nodes2 = [''] * num_nodes
    visited_nodes3 = []
    recursion_stack = [False] * num_nodes
    recursion_stack2 = [''] * num_nodes
    recursion_stack3 = []

    for current_node in range(num_nodes):
        # current_node_name = index2article_with_references[current_node]
        if not visited_nodes[current_node]:
            # print("Running DFS on Art.", current_node_name)
            DFS(current_node, visited_nodes, visited_nodes2, visited_nodes3, recursion_stack, recursion_stack2,
                recursion_stack3, graph_matrix,
                index2article_with_references)


def DFS_kosaraju(graph_matrix, v, visited, visited2, visited3, stack, stack2, index2article_with_references):
    current_node_name = index2article_with_references[v]
    visited[v] = True
    visited2[v] = current_node_name
    visited3.append(current_node_name)
    for adjacent_node in range(len(graph_matrix)):
        if graph_matrix[v][adjacent_node] == 1 and not visited[adjacent_node]:
            DFS_kosaraju(graph_matrix, adjacent_node, visited, visited2, visited3, stack, stack2,
                         index2article_with_references)
    stack.append(v)
    stack2.append(current_node_name)


def DFS_util(transposed_graph_matrix, v, visited, visited2, visited3, component, component2,
             index2article_with_references):
    current_node_name = index2article_with_references[v]
    visited[v] = True
    visited2[v] = current_node_name
    visited3.append(current_node_name)
    component.append(v)
    component2.append(current_node_name)
    for adjacent_node in range(len(transposed_graph_matrix)):
        adjacent_node_name = index2article_with_references[adjacent_node]
        if transposed_graph_matrix[v][adjacent_node] == 1 and not visited[adjacent_node]:
            DFS_util(transposed_graph_matrix, adjacent_node, visited, visited2, visited3, component, component2,
                     index2article_with_references)


def kosaraju_scc(graph_matrix, index2article_with_references):
    stack = []
    stack2 = []
    visited = [False] * len(graph_matrix)
    visited2 = [''] * len(graph_matrix)
    visited3 = []

    for i in range(len(graph_matrix)):
        if not visited[i]:
            DFS_kosaraju(graph_matrix, i, visited, visited2, visited3, stack, stack2, index2article_with_references)

    transposed_graph_matrix = graph_matrix.transpose()

    visited = [False] * len(graph_matrix)
    visited2 = [''] * len(graph_matrix)
    visited3 = []
    scc_list = []

    while stack:
        i = stack.pop()
        if not visited[i]:
            component = []
            component2 = []
            DFS_util(transposed_graph_matrix, i, visited, visited2, visited3, component, component2,
                     index2article_with_references)
            scc_list.append(component)

    return scc_list


def main():
    index2references: dict[int, list] = {}
    index2article_with_references: dict[int, LegalText] = {}
    article2index: dict[str, int] = {}
    article2references: dict[str, list] = {}
    df = pd.read_excel('GDPR_map.xlsx',
                       names=['Article', 'Paragraph', 'Point', 'Translated', 'References', 'Link'],
                       dtype={'Article': str, 'Paragraph': str, 'Point': str, 'Translated': bool, 'References': str,
                              'Link': str})
    df['Link'] = setup_links()
    df.fillna('', inplace=True)
    # print(df)
    setup_dictionaries(df, index2references, index2article_with_references, article2index, article2references)
    # print(index2references)
    # print(index2article_with_references)# maps the number node to correct name
    # print(article2index)
    # print(article2references)

    # choose an option between 0, 1, 2, 3 to set up the graph adjacency matrix
    graph_matrix = setup_graph_matrix(index2article_with_references, article2index, option=1)

    # with np.printoptions(threshold=np.inf):
    # print(graph_matrix)

    detect_cycle(graph_matrix, index2article_with_references)
    if num_cycles > 0:
        print(f"{num_cycles} cycle(s) detected.\n")
    else:
        print("No cycle detected.\n")

    scc_list = kosaraju_scc(graph_matrix, index2article_with_references)
    for i in range(len(scc_list)):
        if len(scc_list[i]) > 1:
            print(f"SCC {i + 1}: {[index2article_with_references[node].name for node in scc_list[i]]}")
            print("Number of nodes in the strongly connected component:", len(scc_list[i]))
    print("Number of strongly connected components including more than one node:",
          len([scc for scc in scc_list if len(scc) > 1]))
    print("Total number of strongly connected components:", len(scc_list))

    # Author: Esteban Garcia Taquez

    # turn the graph adjacency matrix into 2d array
    arrayMatrix = np.array(graph_matrix)

    # this will loop through the matrix and print it
    # for x in graph_matrix:
    #     print(x)

    # turn the 2d array into a matrix
    matrix = np.matrix(arrayMatrix)

    # loop through matrix and write into a txt file
    with open('matrix.txt', 'wb') as f:
        for line in matrix:
            np.savetxt(f, line, fmt='%.0f')

    # create the graph from the matrix as an array
    G = nx.DiGraph(arrayMatrix)

    # modify the names of the nodes
    G = nx.relabel_nodes(G, index2article_with_references)

    # draw and print the graph
    nx.draw(G, pos=nx.circular_layout(G), with_labels=True, node_size=1200)
    # plt.show()


if __name__ == '__main__':
    main()
