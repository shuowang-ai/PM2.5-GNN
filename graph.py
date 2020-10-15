import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham


city_fp = os.path.join(proj_dir, 'data/city.txt')
altitude_fp = os.path.join(proj_dir, 'data/altitude.npy')


class Graph():
    def __init__(self):
        self.dist_thres = 3
        self.alti_thres = 1200
        self.use_altitude = True

        self.altitude = self._load_altitude()
        self.nodes = self._gen_nodes()
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)
        self.edge_index, self.edge_attr = self._gen_edges()
        if self.use_altitude:
            self._update_edges()
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]

    def _load_altitude(self):
        assert os.path.isfile(altitude_fp)
        altitude = np.load(altitude_fp)
        return altitude

    def _lonlat2xy(self, lon, lat, is_aliti):
        if is_aliti:
            lon_l = 100.0
            lon_r = 128.0
            lat_u = 48.0
            lat_d = 16.0
            res = 0.05
        else:
            lon_l = 103.0
            lon_r = 122.0
            lat_u = 42.0
            lat_d = 28.0
            res = 0.125
        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y

    def _gen_nodes(self):
        nodes = OrderedDict()
        with open(city_fp, 'r') as f:
            for line in f:
                idx, city, lon, lat = line.rstrip('\n').split(' ')
                idx = int(idx)
                lon, lat = float(lon), float(lat)
                x, y = self._lonlat2xy(lon, lat, True)
                altitude = self.altitude[y, x]
                nodes.update({idx: {'city': city, 'altitude': altitude, 'lon': lon, 'lat': lat}})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['city']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def gen_lines(self):

        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))

        return lines

    def _gen_edges(self):
        coords = []
        lonlat = {}
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        dist = distance.cdist(coords, coords, 'euclidean')
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.dist_thres] = 1
        assert adj.shape == dist.shape
        dist = dist * adj
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()

        direc_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon

            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direc = mpcalc.wind_direction(u, v)._magnitude

            direc_arr.append(direc)
            dist_kilometer.append(dist_km)

        direc_arr = np.stack(direc_arr)
        dist_arr = np.stack(dist_kilometer)
        attr = np.stack([dist_arr, direc_arr], axis=-1)

        return edge_index, attr

    def _update_edges(self):
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1,0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]
            if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:
                edge_index.append(self.edge_index[:,i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)


if __name__ == '__main__':
    graph = Graph()