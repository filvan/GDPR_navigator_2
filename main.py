import openpyxl
import pandas as pd
import numpy as np
import re
import networkx as nx


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


def setup_graph_matrix(index2article_with_references: dict, article2index: dict, option=0) -> np.ndarray:
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
                    reference_index = article2index[match1]
                    graph_matrix[i][reference_index] = 1
                else:
                    match2 = re.search(pattern=regex2, string=reference)
                    if match2:
                        match2 = match2.string
                        match2 = re.sub(r"[^\d+\-\d+]", "", match2)
                        match2 = match2.split("-")
                        if option == 1:
                            increment = 0
                            interval_start = article2index[match2[0]]
                            interval_end = article2index[match2[1]]
                            for k in range(interval_end, len(index2article_with_references)):
                                if match2[1] in index2article_with_references[k].name:
                                    increment += 1
                        else:
                            increment = 1
                            interval_start = int(match2[0])
                            interval_end = int(match2[1])
                        for j in range(interval_start, interval_end + increment):
                            if option == 1:
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


def DFS(current_node, visited_nodes, recursion_stack, graph_matrix, index2article_with_references):
    current_node_name = index2article_with_references[current_node]
    visited_nodes[current_node] = True
    recursion_stack[current_node] = True
    # print("Visiting Art.", current_node_name)

    for adjacent_node in range(len(index2article_with_references)):
        adjacent_node_name = index2article_with_references[adjacent_node]
        if graph_matrix[current_node][adjacent_node] == 1:
            # print("Art.", current_node_name, "-> Art.", adjacent_node_name)
            if not visited_nodes[adjacent_node]:
                # print("Art.", current_node_name, "-> Art.", adjacent_node_name)
                if DFS(adjacent_node, visited_nodes, recursion_stack, graph_matrix, index2article_with_references):
                    return True
            elif recursion_stack[adjacent_node]:
                print("\nCycle detected between Art.", current_node_name, "and Art.", adjacent_node_name, "\n")
                return True

    recursion_stack[current_node] = False
    # print("Visit on Art.", current_node_name, "completed.")
    return False


def detect_cycle(graph_matrix, index2article_with_references):
    num_nodes = len(index2article_with_references)
    num_cycles = 0

    visited_nodes = [False] * num_nodes
    recursion_stack = [False] * num_nodes

    for current_node in range(num_nodes):
        current_node_name = index2article_with_references[current_node]
        if not visited_nodes[current_node]:
            # print("Running DFS on Art.", current_node_name)
            if DFS(current_node, visited_nodes, recursion_stack, graph_matrix, index2article_with_references):
                num_cycles += 1

    return num_cycles


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
    print(df)
    setup_dictionaries(df, index2references, index2article_with_references, article2index, article2references)
    # print(index2references)
    # print(index2article_with_references)
    # print(article2index)
    # print(article2references)
    graph_matrix = setup_graph_matrix(index2article_with_references, article2index, option=1)
    # with np.printoptions(threshold=np.inf):
    # print(graph_matrix)
    num_cycles = detect_cycle(graph_matrix, index2article_with_references)
    if num_cycles > 0:
        print(f"{num_cycles} cycle(s) detected.")
    else:
        print("No cycle detected.")


if __name__ == '__main__':
    main()
