# Author: Vikram Bhagavatula
# Date: 2022-05-06
# Description: Implementation of siminet algorithm in Python (second version)
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union
from copy import deepcopy


def graph_compare(gcmp, gref, ins_cost, sub_rad, eq_rad,  transform=False):
    """
    Intakes two graphs, gcmp and gref, and computes the node and edge distances b/w them as per the Siminet algorithm.
    transform will transform a copy of gcmp during the course of the function if set to true (for testing purposes).
    """
    
    if transform: 
        copy = deepcopy(gcmp) # ensures that we don't mutate what was passed in
    
    dist = lambda p,q: np.linalg.norm(p[1]["position"] - q[1]["position"]) # compute the Euclidean distance b/w nodes

    equivalency_mapping = dict()
    counterparts = set() # set of all nodes in Gref with counterparts in Gcmp (through substitution or equality)
    
    node_score = 0
    edge_score = 0

    # Node Score
    
    for n in gcmp.nodes(data=True):
        closest = min(gref.nodes(data=True), key=lambda m: dist(m, n), default=np.array([np.inf,np.inf,np.inf])) # if gref is empty, default value returned is node at infinity
        d = dist(n, closest)
        print(f"{d=}")

        if d <= eq_rad: # Equivalency
            equivalency_mapping[closest[0]] = n[0]
            counterparts.add(closest[0])
            if transform:
                copy.nodes[n[0]]["position"] = closest[1]["position"]
        
        elif d <= sub_rad: # Substitution
            # equivalency_mapping[n] = closest
            counterparts.add(closest[0])
            node_score += d
            if transform:
                copy.nodes[n[0]]["position"] = closest[1]["position"]
        
        else: # Deletion
            node_score += ins_cost
            if transform:
                copy.remove_node(n[0])

    not_found = gref.nodes - counterparts # nodes in Gref that had no counterpart (equivalency or substitution) in Gcmp
    node_score += ins_cost * len(not_found) # total insertion cost for nodes not found

    if transform:
        for n in not_found:
            copy.add_node(n, **gref.nodes[n])

    # Edge Score

    for e in gref.edges(data=True):
        n1, n2, ref_data = e
        wt = 0
        add_edge = False
        
        if n1 in equivalency_mapping and n2 in equivalency_mapping:
            pce = (equivalency_mapping[n1], equivalency_mapping[n2]) # finds (potential) corresponding edge based on equivalency mapping

            if pce in gcmp.edges:
                wt = gcmp.edges[pce]["weight"]
            else:
                add_edge = True
        else:
            add_edge = True

        edge_score += abs(ref_data["weight"] - wt)
        
        if transform and add_edge:
            print(f"adding edge b/w {n1} and {n2}")
            copy.add_edge(n1, n2, **ref_data)

    
    if transform:
        return (node_score, edge_score, copy)
    
    return (node_score, edge_score)
