import random
import torch
import networkx as nx

def get_flag():
    pass

# Set common shapes:
house = nx.house_graph()
house_x = nx.house_x_graph()
diamond = nx.diamond_graph()
trian = nx.cycle_graph(n=3)
pentagon = nx.cycle_graph(n=4)
wheel = nx.wheel_graph(n=7)
wheel6 = nx.wheel_graph(n=7)
wheel5 = nx.wheel_graph(n=6)
wheel4 = nx.wheel_graph(n=5)
wheel3 = nx.wheel_graph(n=4)
star = nx.star_graph(n=4)
flag = None

triangle = nx.Graph()
triangle.add_nodes_from([0, 1, 2])
triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])


def random_shape(n) -> nx.Graph:
    '''
    Outputs a random shape as nx.Graph

    ..note:: set `random.seed()` for seeding
    
    Args:
        n (int): Number of shapes in the bank to draw from
    
    '''
    shape_list = [
        house,
        pentagon,
        wheel
    ]
    i = random.choice(list(range(len(shape_list))))
    return shape_list[i], i + 1

