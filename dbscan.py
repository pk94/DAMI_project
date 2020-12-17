import pandas as pd
import math
import operator
import matplotlib.pyplot as plt
import numpy as np


class DBSCAN():
    def __init__(self, dataset, eps=0.3, min_points=5):
        self.dataset = self.load_dataset(dataset)
        self.eps = eps
        self.min_points = min_points
        self.clusters = []
        self.find_neighbours()
        self.clustering()

    def load_dataset(self, dataset):
        return sorted([Point(coordinates) for coordinates in dataset], key=operator.attrgetter('distance'))

    def find_neighbours(self):
        for source_point_id, source_point in enumerate(self.dataset):
            source_point.id = source_point_id
            for point_id, point in enumerate(self.dataset[source_point_id + 1:]):
                if point.distance - source_point.distance < self.eps:
                    distance = math.sqrt(sum([(source_coor - neighbour_coor) ** 2 for source_coor, neighbour_coor
                                              in zip(source_point.coordinates, point.coordinates)]))
                    if distance < self.eps:
                        source_point.add_neighbour(point_id + source_point_id + 1)
                        point.add_neighbour(source_point_id)
                else:
                    break


    def clustering(self):
        for point in self.dataset:
            if len(point.neighbours) + 1 >= self.min_points:
                point.is_core = True
                if len(point.clusters) == 0:
                    current_cluster = Cluster(len(self.clusters))
                    self.clusters.append(current_cluster)
                    current_cluster.add_poind(point.id)
                    point.clusters.append(current_cluster.id)
                    seeds = point.neighbours.copy()
                    point.add_neighbour(point.id)
                    used_seeds = [point.id]
                    while seeds:
                        current_seed = seeds.pop(0)
                        used_seeds.append(current_seed)
                        current_seed = self.dataset[current_seed]
                        current_cluster.add_poind(current_seed.id)
                        current_seed.clusters.append(current_cluster.id)
                        if len(current_seed.neighbours) + 1 >= self.min_points:
                            new_seeds = current_seed.neighbours.copy()
                            seeds = list(set(seeds) | set(new_seeds))
                            for used_seed in used_seeds:
                                if used_seed in seeds:
                                    seeds.remove(used_seed)
                else:
                    pass
        outliner_cluster = Cluster(len(self.clusters))
        self.clusters.append(outliner_cluster)
        for point in self.dataset:
            if len(point.clusters) == 0:
                point.clusters.append(outliner_cluster.id)
                outliner_cluster.add_poind(point.id)

    def plot2d_clusters(self):
        for cluster in self.clusters:
            x = [self.dataset[point].coordinates[0] for point in cluster.points]
            y = [self.dataset[point].coordinates[1] for point in cluster.points]
            if cluster.id == len(self.clusters) - 1:
                plt.scatter(x, y, c='black')
            else:
                plt.scatter(x, y)
        plt.show()




class Cluster():
    def __init__(self, cluster_id):
        self.id = cluster_id
        self.points = []

    def add_poind(self, point):
        self.points.append(point)



class Point():
    def __init__(self, coordinates):
        self.id = 0
        self.coordinates = coordinates
        self.distance = math.sqrt(sum([coor ** 2 for coor in self.coordinates]))
        self.neighbours = []
        self.is_core = False
        self.clusters = []

    def add_neighbour(self, neighbour_idx):
        self.neighbours.append(neighbour_idx)


def load_test_data(path):
    data = pd.read_csv(path, names=['x', 'y', 'class'])
    points = [(row['x'], row['y']) for idx, row in data.iterrows()]
    return points


def list_diff(li1, li2):
    return (list(list(set(li1) - set(li2)) + list(set(li2) - set(li1))))

def RVV_interval(vec_len, epsilon):
    alpha_1 = 1 + 1 / epsilon
    alpha_2 = np.sqrt(alpha_1**2 - 4)
    alpha = (alpha_1 + alpha_2) / 2
    lower_bond = vec_len / alpha
    upper_bond = vec_len * alpha
    return (lower_bond, upper_bond)

def ZPN_interval(vec_len, epsilon):
    lower_bond = vec_len * np.sqrt(epsilon)
    upper_bond = vec_len / np.sqrt(epsilon)
    return (lower_bond, upper_bond)

print(ZPN_interval(10.20, 0.85))
dbscan = DBSCAN(load_test_data('datasets/test_data_2.csv'))
dbscan.plot2d_clusters()
