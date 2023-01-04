import numpy as np
import scipy as sp
import scipy.sparse
import maxflow


class GraphCut(object):
    def __init__(self, nodes, connections):
        """
        nodes: expected number of nodes in the graph
        connections: expected number of edges in the graph
        """
        self._g = maxflow.Graph[float](nodes, connections)
        self._N = nodes
        self._g.add_nodes(self._N)

        self._was_minimized = False
        self._prev_u = np.zeros((self._N, 2))

    def set_neighbors(self, adj_matrix):
        """
        adj_matrix: (N, N) Adjacency matrix in a SciPy sparse format.
            Which one does not matter as it will be converted to coo format.
        """
        adj_m = sp.sparse.triu(adj_matrix.tocoo())

        for i, j, w in zip(adj_m.row, adj_m.col, adj_m.data):
            self._g.add_edge(i, j, w, w)

    def set_unary(self, unaries):
        """
        unaries: (N, 2) array containing the unary weights.
        """
        if not self._was_minimized:
            for i, u in enumerate(unaries):
                self._g.add_tedge(i, u[1], u[0])
        else:
            diff = unaries - self._prev_u
            for i, u in enumerate(diff):
                self._g.add_tedge(i, u[1], u[0])

        self._prev_u[:] = unaries

    def set_pairwise(self, pairwise):
        """
        :param pairwise: (E, 6) array.
                         E: number of non-terminal edges
                         Each row has the format:[i,j,e00,e01,e10,e11], where i and j are neighbours
                         and the four coefficients define the interaction potential.
        """
        for i, p in enumerate(pairwise):
            self.__add_pairwise(p)

    def minimize(self):
        e = self._g.maxflow()
        self._was_minimized = True

        return e

    def get_labeling(self):
        labels = self._g.get_grid_segments(np.arange(self._N))

        return labels

    def __add_pairwise(self, p):
        i, j, e00, e01, e10, e11 = p
        self._g.add_tedge(i, e11, e00)
        e01 -= e00
        e10 -= e11

        # assert e01+ e10 >= 0
        if e01 < 0:
            self._g.add_tedge(i, 0, e01)
            self._g.add_tedge(j, 0, -e01)
            self._g.add_edge(i, j, 0, e01 + e10)
        elif e10 < 0:
            self._g.add_tedge(i, 0, -e10)
            self._g.add_tedge(j, 0, e10)
            self._g.add_edge(i, j, e01 + e10, 0)
        else:
            self._g.add_edge(i, j, e01, e10)
