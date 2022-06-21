# Author: Vikram Bhagavatula
# Date: 2022-05-10
# Description: Tests for siminet algorithm implementation(s)

import siminet as sn
import networkx as nx
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import partial, update_wrapper


def gen_constants(eq_rad):
    """ Generates a set of constants based on the given equivalency radius."""
    constants = {'ins_cost': 1,
                 'eq_rad': eq_rad,
                 'sub_rad': 1.5*eq_rad}

    return constants

def one_empty(impl, constants):
    empty = nx.Graph()
    nonempty = nx.Graph()
    
    # nonempty: 1 -> 2, with weight 1
    nonempty.add_node(1, position=np.array([0,0,0]))
    nonempty.add_node(2, position=np.array([1,0,0]))
    nonempty.add_edge(1, 2, weight=1)

    expected = (2*constants["ins_cost"], nonempty.edges[1,2]["weight"]) # (insertion cost for 2 nodes, abs(weight of edge))
    
    # print(f"{impl(empty, nonempty, **constants)=}")
    return {'gcmp': empty,
            'gref': nonempty,
            'result': impl(deepcopy(empty), nonempty, **constants)
            }
    # assert impl(gcmp=empty, gref=nonempty, **constants) == expected

def same(impl, constants):
    nonempty = nx.Graph()
    
    # nonempty: 1 -> 2, with weight 1
    nonempty.add_node(1, position=np.array([0,0,0]))
    nonempty.add_node(2, position=np.array([1,0,0]))
    nonempty.add_edge(1, 2, weight=1)

    print(f"{impl(nonempty, nonempty, **constants)=}")
    assert impl(nonempty, nonempty, **constants) == (0,0)
    
    return {'gcmp': nonempty,
            'gref': nonempty,
            'result': impl(deepcopy(nonempty), nonempty, **constants)
            }

def upwards(impl, constants, length=10, translation_type='eq'):
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
    
    dir = np.random.rand(2)
    print(f"{dir=}")
    # dir = np.array([1,1,1], dtype='float64') # direction vector for line
    gref = nx.path_graph(length)

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
    horizontal = np.zeros(dir.size)
    horizontal[0] = dir[0]
    offset = horizontal - (np.dot(horizontal, dir) / np.dot(dir, dir)) * dir
    # offset = np.cross(np.array([1,0]), dir) # scales a vector perpendicular to the direction vec and [1,0]
    # offset /= np.linalg.norm(offset)
    offset *= k / np.linalg.norm(offset) # now set to vector that is orthogonal to dir, with magnitude k
    
    print(f"{offset=}")
    print(f"{np.dot(offset, dir)=}")

    gnew = deepcopy(gref)
    for n in gnew.nodes:
        gnew.nodes[n]["position"] += offset

    # breakpoint()
        

    expected = {'eq': 0, # no cost
                'sub': np.linalg.norm(offset) * len(gref.nodes), # Use Euclidean distance for substitution cost
                'out': 2 * constants['ins_cost'] * len(gref.nodes), # insertion and deletion occurs, so 2x the cost
               }

    result = impl(gnew, gref, **constants)
    print(f"{result=}")
    print(f"{expected[translation_type]=}")
    # assert impl(gnew, gref, **constants) == (expected[translation], 0)

    if len(result) == 3: # means we also transformed a graph along the way
        gref_positions = nx.get_node_attributes(gref, "position")
        gnew_positions = nx.get_node_attributes(gnew, "position")
        res_positions = nx.get_node_attributes(result[-1], "position")
        
        # TODO: Plot Gref, Gcmp, and Result with labels
        nx.draw(gref, gref_positions, node_color="r")
        plt.plot()
        plt.title("Gref")
        
        nx.draw(gnew, gnew_positions, node_color='g')
        plt.plot()
        #plt.title("Gcmp")
        # plt.show()
        
        nx.draw(result[-1], res_positions, node_color='b', node_size=150)
        plt.plot()
        #plt.title("Result")
        plt.show()
        
    
    return {'gcmp': gnew,
            'gref': gref,
            'result': result
            }
    # assert impl(gcmp=empty, gref=nonempty, **constants) == expected

# def bigger(impl, constants):
#     g = None
#     assert True

def test_results(testname, res):
    header = testname + '\n' + '-' * (2*len(testname))
    body = '\n'.join(f"  - {k}: {v}" for (k, v) in res.items())

    return header + '\n' + body
    
def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def main():
    impl = sn.graph_compare
    constants = gen_constants(0.5)

    tests = [one_empty,
			 same,
			 wrapped_partial(upwards, translation_type='eq'),
			 wrapped_partial(upwards, translation_type='sub'),
			 wrapped_partial(upwards, translation_type='out'),
            ]

    results = map(lambda t: (t.__name__, t(impl, constants)), tests)
    report = '\n'.join(test_results(*args) for args in results)

    print(report)

if __name__=="__main__":
    main()
