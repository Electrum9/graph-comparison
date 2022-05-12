# Author: Vikram Bhagavatula
# Date: 2022-05-10
# Description: Tests for siminet algorithm implementation(s)

import siminet as sn
import networkx as nx
import numpy as np
from copy import deepcopy


def one_empty(impl, constants):
    empty = nx.Graph()
    nonempty = nx.Graph()
    
    # nonempty: 1 -> 2, with weight 1
    nonempty.add_node(1, position=np.array([0,0,0]))
    nonempty.add_node(2, position=np.array([1,0,0]))
    nonempty.add_edge(1, 2, weight=1)

    expected = (2*constants["ins_cost"], nonempty.edges[1,2]["weight"]) # (insertion cost for 2 nodes, abs(weight of edge))
    
    print(f"{impl(empty, nonempty, **constants)=}")
    assert impl(gcmp=empty, gref=nonempty, **constants) == expected

def same(impl, constants):
    nonempty = nx.Graph()
    
    # nonempty: 1 -> 2, with weight 1
    nonempty.add_node(1, position=np.array([0,0,0]))
    nonempty.add_node(2, position=np.array([1,0,0]))
    nonempty.add_edge(1, 2, weight=1)

    print(f"{impl(nonempty, nonempty, **constants)=}")
    assert impl(nonempty, nonempty, **constants) == (0,0)

def upwards(impl, constants, translation_type='eq'):
    """
    Subjects implementation to a path graph consisting of nodes that lie along a line in space,
    and a similar graph lying along a parallel line in space. Based on the string passed to
    translation_type, we will translate the similar graph so all the nodes are within the equivalence
    radius, substitution radius, or completely outside ('eq', 'sub', 'out'). 

    Depending on the translation_type type, we will expect to see a certain node score, which is either:
        - 'eq': Completely zero.
        - 'sub': Substitution score * number of nodes.
        - 'out': Insertion score * number of nodes.
    
    The edges are all of the same lengths in both graphs, for now.
    """
    
    dir = np.random.rand(3)
    print(f"{dir=}")
    # dir = np.array([1,1,1], dtype='float64') # direction vector for line
    gref = nx.path_graph(10)

    for (i, n) in enumerate(gref.nodes):
        gref.nodes[n]["position"] = i*dir # scale direction vector by some amount

        if n != len(gref.nodes) - 1: # means we're not at end of graph
            gref.edges[n, n+1]["weight"] = 1

    # k = constants.get(translation_type+"_rad", 2*constants['sub_rad'])

    k_map = {'eq': 0.8*constants['eq_rad'], # 80% equivalence radius
             'sub': 0.5*(constants['sub_rad'] + constants['eq_rad']), # average of substitution radius and equivalence radius
             'out': 2*constants['sub_rad'],
             }

    k = k_map[translation_type]
    
    print(f"{k=}")
    offset = np.cross(np.array([1,0,0]), dir) # scales a vector perpendicular to the direction vec and [1,0,0]
    offset /= np.linalg.norm(offset)
    offset *= k
    
    print(f"{offset=}")

    gnew = deepcopy(gref)
    for n in gnew.nodes:
        gnew.nodes[n]["position"] += offset

    # breakpoint()
        

    expected = {'eq': 0,
                'sub': np.linalg.norm(offset) * len(gref.nodes),
                'out': constants['ins_cost'] * len(gref.nodes),
               }

    print(f"{impl(gnew, gref, **constants)=}")
    print(f"{expected[translation_type]=}")
    # assert impl(gnew, gref, **constants) == (expected[translation], 0)

def bigger(impl, constants):
    g = None
    assert True

def main():
    pass

if __name__=="__main__":
    main()
