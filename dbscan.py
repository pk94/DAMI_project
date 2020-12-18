import pandas as pd
import operator
import matplotlib.pyplot as plt
import numpy as np
import abc
import time


class DBSCAN(abc.ABC):
    def __init__(self, dataset, eps=0.3, min_points=5, out_path='out.csv', stats_path='stats.txt'):
        start_time = time.time()
        self.dataset = self.load_dataset(dataset)
        self.filename = ''
        self.eps = eps
        self.min_points = min_points
        self.clusters = []
        self.runtimes = {}
        self.find_neighbours()
        self.clustering()
        self.runtimes.update({'total_runtime': time.time() - start_time})
        self.out_dataframe = self.save_output(out_path)
        self.save_stats(stats_path)

    def load_dataset(self, dataset):
        return sorted([Point(coordinates, unsorted_id) for unsorted_id, coordinates in enumerate(dataset)],
                      key=operator.attrgetter('distance'))

    @abc.abstractmethod
    def neighbourhood_interval(self, vec_len):
        pass

    def tanimoto_similarity(self, vec_1, vec_2):
        dot_product = np.dot(vec_1.coordinates, vec_2.coordinates)
        return 1 - dot_product / (vec_1.distance**2 + vec_2.distance**2 - dot_product)

    def find_neighbours(self):
        start_time = time.time()
        for source_point_id, source_point in enumerate(self.dataset):
            source_point.id = source_point_id
            interval = self.neighbourhood_interval(source_point.distance)
            for point_id, point in enumerate(self.dataset[source_point_id + 1:]):
                if interval[0] <= point.distance <= interval[1]:
                    distance = self.tanimoto_similarity(source_point, point)
                    source_point.sim_calc_num += 1
                    if distance < self.eps:
                        source_point.add_neighbour(point_id + source_point_id + 1)
                        point.add_neighbour(source_point_id)
                else:
                    break
        self.runtimes.update({'neigh_sel_time': time.time() - start_time})

    def clustering(self):
        start_time = time.time()
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
        noise_cluster = Cluster(len(self.clusters))
        self.clusters.append(noise_cluster)
        for point in self.dataset:
            if len(point.clusters) == 0:
                point.is_noise = True
                point.clusters.append(noise_cluster.id)
                noise_cluster.add_poind(point.id)
        self.runtimes.update({'clustering_time': time.time() - start_time})

    def save_output(self, savepath):
        dim_columns = [f'dim_{idx}' for idx in range(len(self.dataset[0].coordinates))]
        out_dataframe = pd.DataFrame(columns=['id'] + dim_columns + ['sim_calc_num', 'point_type', 'cluster_id'])
        for point in self.dataset:
            if point.is_core:
                point_type = 1
            elif not point.is_core and not point.is_noise:
                point_type = 0
            else:
                point_type = -1
            row_dict = {dim: value for dim, value in zip(dim_columns, point.coordinates)}
            row_dict.update({'id': point.unsorted_id, 'sim_calc_num': point.sim_calc_num, 'point_type': point_type,
                             'cluster_id': point.clusters})
            out_dataframe = out_dataframe.append(row_dict, ignore_index=True)
        out_dataframe.to_csv(savepath)
        return out_dataframe

    def save_stats(self, savepath):
        with open(savepath, 'w') as file:
            file.write(f'input file name: {self.filename}, dimensions number: {len(self.dataset[0].coordinates)}, '
                       f'number of points: {len(self.dataset)}\n')
            file.write(f'epsilor: {self.eps}, minimum number of points {self.min_points}\n')
            file.write(f'neighbourhood selection time: {self.runtimes["neigh_sel_time"]}, '
                       f' clustering time: {self.runtimes["clustering_time"]}, '
                       f' total runtime: {self.runtimes["total_runtime"]}\n')
            file.write(f'num of discovered clusters: {len(self.clusters) - 1}, '
                       f'num of core points: {self.out_dataframe[self.out_dataframe["point_type"] == 1].shape[0]}, '
                       f'num of border points: {self.out_dataframe[self.out_dataframe["point_type"] == 0].shape[0]}, '
                       f'num of noise points: {self.out_dataframe[self.out_dataframe["point_type"] == -1].shape[0]}\n')
            file.write(f'average number of distance caluclations for point: {self.out_dataframe["sim_calc_num"].mean()}')



    def plot2d_clusters(self):
        for cluster in self.clusters:
            x = [self.dataset[point].coordinates[0] for point in cluster.points]
            y = [self.dataset[point].coordinates[1] for point in cluster.points]
            if cluster.id == len(self.clusters) - 1:
                plt.scatter(x, y, c='black')
            else:
                plt.scatter(x, y)
        plt.show()


class ClassicDBSCAN(DBSCAN):
    def __init__(self, dataset, eps=0.3, min_points=5):
        super(ClassicDBSCAN, self).__init__(dataset, eps, min_points)

    def neighbourhood_interval(self, vec_len):
        lower_bond = vec_len - self.eps
        upper_bond = vec_len + self.eps
        return (lower_bond, upper_bond)


class ZPNDBSCAN(DBSCAN):
    def __init__(self, dataset, eps=0.3, min_points=5):
        super(ZPNDBSCAN, self).__init__(dataset, eps, min_points)

    def neighbourhood_interval(self, vec_len):
        lower_bond = vec_len * np.sqrt(self.eps)
        upper_bond = vec_len / np.sqrt(self.eps)
        return (lower_bond, upper_bond)


class RVVDBSCAN(DBSCAN):
    def __init__(self, dataset, eps=0.3, min_points=5):
        super(RVVDBSCAN, self).__init__(dataset, eps, min_points)

    def neighbourhood_interval(self, vec_len):
        alpha_1 = 1 + 1 / self.eps
        alpha_2 = np.sqrt(alpha_1 ** 2 - 4)
        alpha = (alpha_1 + alpha_2) / 2
        lower_bond = vec_len / alpha
        upper_bond = vec_len * alpha
        return (lower_bond, upper_bond)


class Cluster():
    def __init__(self, cluster_id):
        self.id = cluster_id
        self.points = []

    def add_poind(self, point):
        self.points.append(point)


class Point():
    def __init__(self, coordinates, unsorted_id):
        self.id = 0
        self.unsorted_id = unsorted_id
        self.coordinates = np.asarray(coordinates)
        self.distance = np.sqrt(sum([coor ** 2 for coor in self.coordinates]))
        self.neighbours = []
        self.is_core = False
        self.is_noise = False
        self.clusters = []
        self.sim_calc_num = 0

    def add_neighbour(self, neighbour_idx):
        self.neighbours.append(neighbour_idx)


def load_test_data(path):
    data = pd.read_csv(path, names=['x', 'y', 'class'])
    points = [(row['x'], row['y']) for idx, row in data.iterrows()]
    return points


dbscan = RVVDBSCAN(load_test_data('datasets/test_data_2.csv'), 0.01, 10)
dbscan.plot2d_clusters()
