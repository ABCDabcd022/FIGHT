import random
import time
from collections import defaultdict

class MaxCliqueTabuSearch:
    def __init__(self):
        self.neighbour_sets = []
        self.non_neighbours = []
        self.best_clique = set()
        self.qco = []
        self.index = []
        self.q_border = 0
        self.c_border = 0

    @staticmethod
    def get_random(a, b):
        return random.randint(a, b)

    def read_graph_file(self, filename):
        with open(filename, 'r') as fin:
            vertices = 0
            edges = 0
            for line in fin:
                if line.startswith('c'):
                    continue
                parts = line.split()
                if parts[0] == 'p':
                    vertices = int(parts[2])
                    edges = int(parts[3])
                    self.neighbour_sets = [set() for _ in range(vertices)]
                    self.qco = [i for i in range(vertices)]
                    self.index = [-1] * vertices
                    self.non_neighbours = [set() for _ in range(vertices)]
                elif parts[0] == 'e':
                    start = int(parts[1]) - 1
                    finish = int(parts[2]) - 1
                    self.neighbour_sets[start].add(finish)
                    self.neighbour_sets[finish].add(start)

            for i in range(vertices):
                for j in range(vertices):
                    if i != j and j not in self.neighbour_sets[i]:
                        self.non_neighbours[i].add(j)

    def run_search(self, starts, randomization):
        for _ in range(starts):
            self.clear_clique()
            for i in range(len(self.neighbour_sets)):
                self.qco[i] = i
                self.index[i] = i
            self.run_initial_heuristic(randomization)
            self.c_border = self.q_border
            swaps = 0
            while swaps < 100:
                if not self.move():
                    if not self.swap_1_to_1():
                        break
                    else:
                        swaps += 1
            if self.q_border > len(self.best_clique):
                self.best_clique.clear()
                for i in range(self.q_border):
                    self.best_clique.add(self.qco[i])

    def get_clique(self):
        return self.best_clique

    def check(self):
        for i in self.best_clique:
            for j in self.best_clique:
                if i != j and j not in self.neighbour_sets[i]:
                    print("Returned subgraph is not clique")
                    return False
        return True

    def clear_clique(self):
        self.q_border = 0
        self.c_border = 0

    def compute_tightness(self, vertex):
        tightness = 0
        for i in range(self.q_border):
            if vertex in self.non_neighbours[self.qco[i]]:
                tightness += 1
        return tightness

    def swap_vertices(self, vertex, border):
        vertex_at_border = self.qco[border]
        self.qco[self.index[vertex]], self.qco[border] = self.qco[border], self.qco[self.index[vertex]]
        self.index[vertex], self.index[vertex_at_border] = self.index[vertex_at_border], self.index[vertex]

    def insert_to_clique(self, i):
        for j in self.non_neighbours[i]:
            if self.compute_tightness(j) == 0:
                self.c_border -= 1
                self.swap_vertices(j, self.c_border)
        self.swap_vertices(i, self.q_border)
        self.q_border += 1

    def remove_from_clique(self, k):
        for j in self.non_neighbours[k]:
            if self.compute_tightness(j) == 1:
                self.swap_vertices(j, self.c_border)
                self.c_border += 1
        self.q_border -= 1
        self.swap_vertices(k, self.q_border)

    def swap_1_to_1(self):
        for counter in range(self.q_border):
            vertex = self.qco[counter]
            for i in self.non_neighbours[vertex]:
                if self.compute_tightness(i) == 1:
                    self.remove_from_clique(vertex)
                    self.insert_to_clique(i)
                    return True
        return False

    def move(self):
        if self.c_border == self.q_border:
            return False
        vertex = self.qco[self.q_border]
        self.insert_to_clique(vertex)
        return True

    def run_initial_heuristic(self, randomization):
        candidates = list(range(len(self.neighbour_sets)))
        random.shuffle(candidates)
        while candidates:
            last = len(candidates) - 1
            rnd = self.get_random(0, min(randomization - 1, last))
            vertex = candidates[rnd]
            self.swap_vertices(vertex, self.q_border)
            self.q_border += 1
            candidates = [c for c in candidates if c in self.neighbour_sets[vertex]]
            random.shuffle(candidates)


def main():
    # iterations = int(input("Number of iterations: "))
    # randomization = int(input("Randomization: "))
    files = {"brock200_1.clq" : [150, 300], "brock200_2.clq" : [150, 300], "brock200_3.clq" : [150, 300], "brock200_4.clq" : [150, 300], 
        "brock400_1.clq" : [300, 200], "brock400_2.clq" : [300, 200], "brock400_3.clq" : [300, 200], "brock400_4.clq" : [300, 200],
        "C125.9.clq" : [100], "gen200_p0.9_44.clq" : [500, 300], "gen200_p0.9_55.clq" : [500], "hamming8-4.clq" : [100], 
        "johnson16-2-4.clq" : [100], "johnson8-2-4.clq" : [100], "keller4.clq" : [100],
        "MANN_a27.clq" : [2000], "MANN_a9.clq" : [2500], 
        "p_hat1000-1.clq": [100, 300], "p_hat1000-2.clq" : [500], "p_hat1500-1.clq" : [100],
        "p_hat300-3.clq" : [300, 300], "p_hat500-3.clq" : [500, 300],  
        "san1000.clq" : [100], "sanr200_0.9.clq" : [500, 300], "sanr400_0.7.clq" : [300]
        }
    with open("clique_tabu.csv", 'w') as fout:
        fout.write("File; Clique; Time (sec)\n")
        for file in files.keys():
            iterations = files[file][0]
            if len(files[file]) > 1:
                randomization = files[file][1]
            else:
                randomization = 100

            problem = MaxCliqueTabuSearch()
            problem.read_graph_file(f"/home/andrey/Projects/HSE/HSE_mag_Data_Mining/Batsyn/data/clique/{file}")
            start = time.time()
            problem.run_search(iterations, randomization)
            if not problem.check():
                print("*** WARNING: incorrect clique ***")
                fout.write("*** WARNING: incorrect clique ***\n")
            fout.write(f"{file}; {len(problem.get_clique())}; {time.time() - start}\n")
            print(f"{file}, result - {len(problem.get_clique())}, time - {time.time() - start}")

if __name__ == "__main__":
    main()