import math
import numpy as np
import networkx as nx
from collections import Counter
from sklearn.linear_model import HuberRegressor


class FeatureExtractor:

    def __init__(self, graph) -> None:
        self.G = graph

    """
    This method returns the depth of the graph. 
    For the calculation we need to pass the networkx graph 
    instance G and the root of the graph, G_root. 
    See [1] for more information about this.
    """

    def calc_depth(self, G_root):
        depth = nx.eccentricity(self.G, v=G_root)
        return depth

    """
    This method defines the size [1] of the graph in question. 
    This simply considers the total number of nodes in the given graph G.
    """

    def calc_size(self):
        return self.G.number_of_nodes()

    """
    Definition needs to be re-visited from [1]
    """

    def max_breadth(self, root, depth):
        
        metadata={}
        node2date = nx.get_node_attributes(self.G, 't_date')
        metadata['depth2breadth'] = {}
        done_nodes = []

        for dp in range(depth):
            current_dp = dp+1
            current_subgraph = nx.ego_graph(self.G, root, radius=current_dp)
            current_sb_nodes = current_subgraph.nodes()
            node2date = nx.get_node_attributes(current_subgraph, 't_date')
            node_dates = node2date.items()
            node_dates = [(n, d) for n, d in node_dates if n not in done_nodes]
            if len(node_dates) == 0:
                break
            current_depth_nodes = [n for n, d in node_dates]

            current_breadth = len(current_depth_nodes)
            metadata['depth2breadth'][current_dp] = current_breadth
            done_nodes = current_sb_nodes

        breadths = metadata['depth2breadth'].values()
        maxbreadth = 1 if len(breadths) == 0 else max(breadths)
        return maxbreadth

    """
    Structural Virality is the modified Wiener index defined in [4]. 
    It is adopted by several authors. Here, we are re-using the definition 
    provided in [1]. Here, tthe graph G needs to be passed down, along with 
    the size of the graph, calculated using calc_size method, defined above.
    """

    def calc_structural_viralty(self, size):
        if size == 1:
            return None  # virality is not defined for cascades of size 1,
        # Note: this is very time-consuming for larger cascades
        sv = nx.average_shortest_path_length(self.G)
        return sv

    """
    This method defines the number of strongly connected components [5, 6] 
    in the given graph. This utilizes the method provided by networkx.
    """

    def no_of_strongly_connnected_components(self):
        return nx.number_strongly_connected_components(self.G)

    """
    This method provides the length of the largest strongly connected 
    component [5, 6]. Using direct method provided by networkx package.
    """

    def size_of_largest_strongly_connected_component(self):
        generator_set = max(nx.strongly_connected_components(self.G), key=len)
        return len(generator_set)

    """
    This method defines the number of weakly connected components [5, 6] 
    in the given graph. This utilizes the method provided by networkx.
    """

    def no_of_weakly_connnected_components(self):
        return nx.number_weakly_connected_components(self.G)

    """
    This method provides the length of the largest weakly connected 
    component [5, 6]. Using direct method provided by networkx package.
    """

    def size_of_largest_weakly_connected_component(self):
        generator_set = max(nx.weakly_connected_components(self.G), key=len)
        return len(generator_set)

    """
    Compute the average clustering coefficient for the graph G.
    Utilizing method provided by networkx package.
    """

    def average_clustering_coefficient(self):
        return nx.average_clustering(self.G)

    """
    The main core is the core with the largest degree.
    A k-core is a maximal subgraph that contains nodes of degree k or more.
    main k core [5, 6] will be core with largest degree.
    """

    def main_k_core_number(self):
        return nx.k_core(self.G)

    def denity(self):
        return nx.density(self.G)

    """
    Calculates structural heterogeneity of a network. It starts by getting the 
    degrees per node and then follows the calculation provided in [3].
    """

    def structural_heterogeneity(self, N):

        squared_sum = 0
        regular_sum = 0

        degree_values = dict(self.G.degree()).values()

        for item in degree_values:
            squared_sum += (item**2)
            regular_sum += item
        
        
        if N != 0 and regular_sum != 0:
            return (math.sqrt(squared_sum/N))/(regular_sum/N)
        else: 
            return 0

    """
    Calculation of characteristic distance, defined in [3]. Calculating distance 
    between all pairs of nodes and then plotting them against their probabilites, 
    calculated as PMF. Then fitting a regression line using robust regression, Huber 
    Regressor, we get the slope of the line, inverse of which is the characteristic 
    distance. For more detail about the calculation, please visit [3].
    """

    def characteristic_distance(self):
        
        distances = []

        length = nx.all_pairs_shortest_path_length(self.G)
        pairs = dict(length)
        
        for n in pairs.keys():
            pair = pairs.get(n)
            for j in pair.keys():
                item = pair.get(j)
                distances.append(item)
        
        
        d6 = Pmf(distances)
        d6.normalize()
        temp_array = []
        
        for item in list(d6.values()):
            try:
                temp_array.append(math.log(item))
            except ValueError:
                return 0
        values = np.array(temp_array).reshape(-1, 1)
        lm = HuberRegressor()
        lm.fit(np.array(list(d6.keys())).reshape(-1, 1), values)

        if lm.coef_ != 0:
            a = -1 * (1/lm.coef_)
        else:
            return -99999

        return a[0]

    def layer_ratio(self, root_node):
        layer_1_ego_graph = nx.ego_graph(self.G, root_node, radius=1)
        layer_2_ego_graph = nx.ego_graph(self.G, root_node, radius=2)
        layer_1_nodes = layer_1_ego_graph.number_of_nodes()
        layer_2_nodes = layer_2_ego_graph.number_of_nodes()
        if layer_1_nodes != 0:
            layer_ratio = layer_2_nodes/layer_1_nodes
        else:
            layer_ratio = 0
        return layer_ratio

    def extract_features(self, root):
        
        depth = self.calc_depth(root)
        size = self.calc_size()
        max_breadth = self.max_breadth(root, depth)
        structural_virality = self.calc_structural_viralty(size)
        strongly_connected_comp = self.no_of_strongly_connnected_components()
        weakly_connected_comp = self.no_of_weakly_connnected_components()
        size_of_largest_weakly_connected_comp = self.size_of_largest_weakly_connected_component()
        avg_clustering_coeff = self.average_clustering_coefficient()
        # main_k_core_no = self.main_k_core_number()
        density = self.denity()
        structural_heterogeneity = self.structural_heterogeneity(size)
        characteristic_distance = self.characteristic_distance()
        layer_ratio = self.layer_ratio(root)

        return (depth, size, max_breadth, structural_virality, 
        strongly_connected_comp, weakly_connected_comp, 
        size_of_largest_weakly_connected_comp, 
        avg_clustering_coeff,  density, layer_ratio,
        structural_heterogeneity, characteristic_distance)


class Pmf(Counter):
    """A Counter with probabilities."""

    def normalize(self):
        """Normalizes the PMF so the probabilities add to 1."""
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total
            self[key] = round(self[key], 3)
