# Author: Vikram Bhagavatula
# Date: 2022-05-06
# Description: Implementation of siminet algorithm in Python (second version)
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union

# @dataclass
# class Substitution:
#     node_id: int
#     direction: np.array
#     # shifts node_id in direction vec
    

# @dataclass
# class InsertNode:
#     position: np.array

# @dataclass
# class InsertEdge:
#     weight: float
#     edge: tuple # pair of node_ids

# @dataclass
# class Deletion:
#     node_id: int

# Instructions = Union[Substitution, InsertNode, InsertEdge, Deletion]

# Transformation constructors: These are functions that accept some arguments, and form a transformation on the given graph.

def substitution(node_id, direction):
    def t(g):
        g.nodes[node_id]["position"] += direction
        return g
    return t

def insert_node(position):
    def t(g):
        node_id = len(g.nodes)
        g.add_node(node_id, position=position)
        return g
    return t

def insert_edge(edge, weight):
    def t(g):
        g.add_edge(edge, weight=weight)
        return g
    return t

def deletion(node_id):
    def t(g):
        g.remove_node(node_id)
        return g
    return t


def graph_compare(gcmp, gref, ins_cost, sub_rad, eq_rad,  trace=False):
    """
    Intakes two graphs, gcmp and gref, and computes the node and edge distances b/w them as per the Siminet algorithm.
    An equivalency is established when trace will track the sequence of operations performed on gcmp in order to transform it into gref, if set to True.
    """
    dist = lambda p,q: np.linalg.norm(p[1]["position"] - q[1]["position"]) # compute the Euclidean distance b/w nodes

    equivalency_mapping = dict()
    counterparts = set() # set of all nodes in Gref with counterparts in Gcmp (through substitution or equality)
    
    if trace: log = [] # set log to empty list if set to true
    
    node_score = 0
    edge_score = 0

    # Node Score
    
    for n in gcmp.nodes(data=True):
        closest = min(gref.nodes(data=True), key=lambda m: dist(m, n), default=np.array([np.inf,np.inf,np.inf])) # if gref is empty, default value returned is node at infinity
        d = dist(n, closest)
        print(f"{d=}")
        operation = None

        if d <= eq_rad: # Equivalency
 #            print(" Equivalency")
            equivalency_mapping[closest[0]] = n[0]
            counterparts.add(closest[0])
        elif d <= sub_rad: # Substitution
 #            print(" Substitution")
            # equivalency_mapping[n] = closest
            counterparts.add(closest[0])
            node_score += d
        else: # Deletion
 #            print(" Deletion")
            node_score += ins_cost

        if trace:
            log.append(operation)

    not_found = gref.nodes - counterparts # nodes in Gref that had no counterpart (equivalency or substitution) in Gcmp
    node_score += ins_cost * len(not_found) # total insertion cost for nodes not found

    # Edge Score

    for e in gref.edges(data=True):
        n1, n2, ref_data = e
        wt = 0
        
        if n1 in equivalency_mapping and n2 in equivalency_mapping:
            pce = (equivalency_mapping[n1], equivalency_mapping[n2]) # finds (potential) corresponding edge based on equivalency mapping
            wt = gcmp.edges[pce]["weight"] if pce in gcmp.edges else 0

        edge_score += abs(ref_data["weight"] - wt)
    
    if trace:
        return (node_score, edge_score, log)
    
    return (node_score, edge_score)

def gen_animation(graph, operations):
    """
    Intakes a graph, and a sequence of operations to perform on it, and synthesizes an animation
    showing how the state of the graph changes with each operation.
    """

    def animate(ops):
        initial = graph
        curr_op = iter(ops)
        
        def a(i):
            nonlocal initial, curr_op
            
            return initial,
        
        return a
    return
