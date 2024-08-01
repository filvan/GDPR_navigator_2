import openpyxl
import pandas as pd
import numpy as np
import re


class LegalText:
    def __init__(self, name, translated=False, references=None):
        self.name = name
        self.translated = translated
        if references is None:
            references = []
        self.references = references


def setup_graph_matrix(index2article_with_references: dict, article2index: dict) -> np.ndarray:
    graph_matrix = np.zeros((len(index2article_with_references) + 1, len(index2article_with_references) + 1), dtype=int)
    for i in range(len(index2article_with_references)):
        graph_matrix[0][i] = i
        graph_matrix[i][0] = i
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
                        candidate_number1 = int(match2[0])
                        candidate_number2 = int(match2[1])
                        for j in range(candidate_number1, candidate_number2 + 1):
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


def setup_dictionaries(df, index2references, index2article_with_references, article2index):
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


def main():
    index2references = {}
    index2article_with_references = {}
    article2index = {}
    df = pd.read_excel('GDPR_map.xlsx',
                       names=['Article', 'Paragraph', 'Point', 'Translated', 'References', 'Link'],
                       dtype={'Article': str, 'Paragraph': str, 'Point': str, 'Translated': bool, 'References': str,
                              'Link': str})
    df['Link'] = setup_links()
    df.fillna('', inplace=True)
    print(df)
    setup_dictionaries(df, index2references, index2article_with_references, article2index)
    print(index2references)
    print(article2index)
    graph_matrix = setup_graph_matrix(index2article_with_references, article2index)
    # with np.printoptions(threshold=np.inf):
    # print(graph_matrix)


if __name__ == '__main__':
    main()
