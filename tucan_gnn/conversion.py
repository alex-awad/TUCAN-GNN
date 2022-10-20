""" Module to convert TUCAN strings into molecular graphs. """

""" Module to convert TUCAN strings to pytorch geometric graphs for their utilization 
    in graph neural networks.
"""

import re
import collections
from tucan.element_properties import element_symbols
from typing import List, Tuple, Dict

# Assign elements to their atomic number
atom_number_dict = {}
number_atom_dict = {}
for i, element in enumerate(element_symbols):
    atom_number_dict[element] = i+1
    number_atom_dict[i+1] = element
    

def element_count_from_sum_formula(sum_form: str)->Dict[str, int]:
    """ Returns elements and their count in a sum formula.
    """
    # Split sum formula on letters and numbers
    split_sum_form = re.split("(\d+)", sum_form)
    if split_sum_form[-1] == "":
        split_sum_form.pop()

    previous_element = ""
    element_count = dict()
    for substring in split_sum_form:
        # Split strings by capital letter, e.g. ClBr => [Cl, Br], "H" => ["H"], 1 => []
        split_by_capital_letter = re.findall('[A-Z][^A-Z]*', substring)
        
        if not split_by_capital_letter:
            element_count[previous_element] = int(substring)
        elif len(split_by_capital_letter) == 1:
            previous_element = split_by_capital_letter[0]
        # Multiple elements in one substring
        else:
            for element in split_by_capital_letter[:-1]:
                element_count[element] = 1
            previous_element = split_by_capital_letter[-1]
    # Check if last element is only present once
    if(split_sum_form[-1].isalpha()):
        element_count[previous_element] = 1
        
    return element_count


def graph_nodes_from_element_count(element_count: dict)->Dict[int, str]:
    """ Returns labels of graph nodes (counting from 1, ascending) and the corresponding element.
    """
    # Change keys of dict to have atomic number as key instead of element abbreviation
    element_count_by_atom_number = dict((atom_number_dict[key], value) \
        for (key, value) in element_count.items())
    
    # Order dict by ascending atom number
    element_count_by_atom_number = collections.OrderedDict(
        sorted(element_count_by_atom_number.items())
    )
    
    # Assign graph nodes
    node_list = []
    for key, value in element_count_by_atom_number.items():
        for i in range(0, value):
            node_list.append(number_atom_dict[key])
            
    graph_nodes = {}
    for i, element in enumerate(node_list):
        graph_nodes[i+1] = element
        
    return graph_nodes


def convert_graph_nodes_element_to_atomic_number(graph_nodes: dict)->Dict:
    """ Converts the keys of graph nodes from the element symbol to the atomic number.
    """
    graph_nodes_converted = dict()
    for node, value in graph_nodes.items():
        graph_nodes_converted[node] = atom_number_dict[value]
        
    return graph_nodes_converted


def tucan_string_to_graph_nodes(tucan_string: str, use_element_symbols: bool = True)->Dict[int, str]:
    """ Transforms a TUCAN string to a dictionary with node labels for graph
        representation. The node labels are enumerated and ordered after the 
        atom number of the elements present in the TUCAN string. 
        
        E.g.: C2H4O => {1: H, 2: H, 3: H, 4: H, 5: C, 6: C, 7: O}
        
        Parameters:
        ------------
        tucan_string (str): TUCAN string from which graph nodes are created.
        
        use_element_symbols (bool): Whether to use element_symbols as node values. Otherwise
            atomic numbers are used. Defaults to True.
    """
    sum_formula = tucan_string.split("/")[0]
    element_counts = element_count_from_sum_formula(sum_formula)
    graph_nodes =  graph_nodes_from_element_count(element_counts)
    
    if use_element_symbols is False:
        graph_nodes = convert_graph_nodes_element_to_atomic_number(graph_nodes)
    
    return graph_nodes


def tucan_string_to_graph_edges(tucan_string: str)->List[Tuple[str]]:
    """ Returns list of edges from TUCAN string as a list for further
        transformation to a graph. Uses the same labels for the atoms as
        tucan_string_to_graph_nodes. 
    """
    edge_section = tucan_string.split("/")[1]
    edges_with_opening_bracket = edge_section.split(")")[:-1]
    edges = [edge[1:] for edge in edges_with_opening_bracket] # Drop opening bracket
    edges_list = [tuple(int(node) for node in edge.split("-")) for edge in edges]
    
    return edges_list


def tucan_string_to_graph(tucan_string: str)->Dict:
    """Returns dict with graph nodes and graph edges from TUCAN string."""
    nodes = tucan_string_to_graph_nodes(tucan_string)
    edges = tucan_string_to_graph_edges(tucan_string)
    
    return {
        "nodes": nodes,
        "edges": edges,
    }
