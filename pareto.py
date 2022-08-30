import pandas as pd
import numpy as np
import matplotlib as mpl
import shapely.geometry as geom

mpl.use('TkAgg')


class ObjectivesSpace:
    def __init__(self, df, functions, path, model):
        self.functions = functions
        self.df = df[df.columns.intersection(self._constr_obj())]
        self.points = self._get_points()
        self.path = path[:-4]
        self.model = model

    def _constr_obj(self):
        objectives = list(self.functions.keys())
        objectives.insert(0, 'model')
        return objectives

    def _get_points(self):
        pts = self.df.to_numpy()
        # pts = obj_pts.copy()
        # obj_pts = obj_pts[obj_pts.sum(1).argsort()[::-1]]
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        pts[:, 1:] = pts[:, 1:] * factors
        # sort points by decreasing sum of coordinates: the point having the greatest sum will be non dominated
        pts = pts[pts[:, 1:].sum(1).argsort()[::-1]]
        # initialize a boolean mask for non dominated and dominated points (in order to be contrastive)
        non_dominated = np.ones(pts.shape[0], dtype=bool)
        dominated = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            # process each point in turn
            n = pts.shape[0]
            # definition of Pareto optimality: for each point in the iteration, we find all points non dominated by
            # that point.
            mask1 = (pts[i + 1:, 1:] >= pts[i, 1:])
            mask2 = np.logical_not(pts[i + 1:, 1:] <= pts[i, 1:])
            non_dominated[i + 1:n] = (np.logical_and(mask1, mask2)).any(1)
            # A point could dominate another point, but it could also be dominated by a previous one in the iteration.
            # The following row take care of this situation by "keeping in memory" all dominated points in previous
            # iterations.
            dominated[i + 1:n] = np.logical_or(np.logical_not(non_dominated[i + 1:n]), dominated[i + 1:n])
        pts[:, 1:] = pts[:, 1:] * factors
        return pts[(np.logical_not(dominated))], pts[dominated]

    @staticmethod
    def _parser(model):
        return float(model.split('_')[5].split("=")[1].replace('$', '.'))

    @staticmethod
    def _parserUltra(model):
        return float(model.split('_')[20].split("=")[1].replace('$', '.'))

    @staticmethod
    def _parserLight(model):
        return float(model.split('_')[9].split("=")[1].replace('$', '.'))

    @staticmethod
    def _parserGCN(model):
        return float(model.split('_')[11].split("=")[1].replace('$', '.'))

    def get_nondominated(self):
        return self.points[0]

    def get_dominated(self):
        return self.points[1]

    def get_distances(self):
        distances = {}
        line = geom.LineString(self.points[0][:, 1:][self.points[0][:, 2].argsort()])
        i = 0
        for point in np.concatenate((self.points[1][:, 1:], self.points[0][:, 1:]), axis=0):
            distances[(i, tuple(point))] = geom.Point(point).distance(line)
            i += 1
        return distances

    def get_statistics(self):
        distances = self.get_distances()
        mean = np.fromiter(distances.values(), dtype=float).mean()
        variance = ((np.fromiter(distances.values(), dtype=float) - mean) ** 2).sum() / (
                    np.fromiter(distances.values(), dtype=float).shape[0] - 1)
        standard_deviation = variance ** (1 / 2)
        return standard_deviation, mean

    def _get_distances_per_hyper(self, hyper_points):
        distances = {}
        line = geom.LineString(self.points[0][:, 1:][self.points[0][:, 2].argsort()])
        i = 0
        for point in hyper_points[:, 1:]:
            distances[(i, tuple(point))] = geom.Point(point).distance(line)
            i += 1
        return distances

    @staticmethod
    def _get_statistics_per_hyper(distances):
        mean = np.fromiter(distances.values(), dtype=float).mean()
        variance = ((np.fromiter(distances.values(), dtype=float) - mean) ** 2).sum() / (
                    np.fromiter(distances.values(), dtype=float).shape[0] - 1)
        standard_deviation = variance ** (1 / 2)
        return standard_deviation, mean

    def get_statistics_per_hyper(self):
        df_points = pd.DataFrame(np.concatenate((self.points[1], self.points[0]), axis=0), columns=['model', 'x', 'y'])
        df_points['factors'] = df_points['model'].map(ObjectivesSpace._parser)
        for el in np.unique(np.array(df_points['factors'])):
            hyper_points = df_points.loc[df_points['factors'] == el].values
            distances = self._get_distances_per_hyper(hyper_points)
            std, mean = self._get_statistics_per_hyper(distances)
            print(f'--STANDARD DEVIATION, MEAN FOR factors={el}')
            print(std, mean)
        if self.model == 'UltraGCN':
            df_points['l'] = df_points['model'].map(ObjectivesSpace._parserUltra)
            for el in np.unique(np.array(df_points['l'])):
                hyper_points = df_points.loc[df_points['l'] == el].values
                distances = self._get_distances_per_hyper(hyper_points)
                std, mean = self._get_statistics_per_hyper(distances)
                print(f'--STANDARD DEVIATION, MEAN FOR l={el}')
                print(std, mean)
        elif self.model == 'LightGCN':
            df_points['layers'] = df_points['model'].map(ObjectivesSpace._parserLight)
            for el in np.unique(np.array(df_points['layers'])):
                hyper_points = df_points.loc[df_points['layers'] == el].values
                distances = self._get_distances_per_hyper(hyper_points)
                std, mean = self._get_statistics_per_hyper(distances)
                print(f'--STANDARD DEVIATION, MEAN FOR layers={el}')
                print(std, mean)
        elif self.model == 'GCN':
            df_points['layers'] = df_points['model'].map(ObjectivesSpace._parserGCN)
            for el in np.unique(np.array(df_points['layers'])):
                hyper_points = df_points.loc[df_points['layers'] == el].values
                distances = self._get_distances_per_hyper(hyper_points)
                std, mean = self._get_statistics_per_hyper(distances)
                print(f'--STANDARD DEVIATION, MEAN FOR layers={el}')
                print(std, mean)

        # df_dominated['layers'] = df_dominated['model'].map(ObjectivesSpace._parserGCN)
        # df_dominated['layers'] = df_dominated['model'].map(ObjectivesSpace._parserLight)


    def to_csv(self):
        df_nondominated = pd.DataFrame(self.points[0], columns=self._constr_obj())
        df_nondominated = df_nondominated.sort_values(by=list(self.functions.keys())[1])
        df_dominated = pd.DataFrame(self.points[1], columns=self._constr_obj())
        df_nondominated['factors'] = df_nondominated['model'].map(ObjectivesSpace._parser)
        if self.model == 'UltraGCN':
            df_nondominated['l'] = df_nondominated['model'].map(ObjectivesSpace._parserUltra)
        elif self.model == 'GCN':
            df_nondominated['layers'] = df_nondominated['model'].map(ObjectivesSpace._parserGCN)
        elif self.model == 'LightGCN':
            df_nondominated['layers'] = df_nondominated['model'].map(ObjectivesSpace._parserLight)
        df_dominated['factors'] = df_dominated['model'].map(ObjectivesSpace._parser)
        if self.model == 'UltraGCN':
            df_dominated['l'] = df_dominated['model'].map(ObjectivesSpace._parserUltra)
        elif self.model == 'GCN':
            df_dominated['layers'] = df_dominated['model'].map(ObjectivesSpace._parserGCN)
        elif self.model == 'LightGCN':
            df_dominated['layers'] = df_dominated['model'].map(ObjectivesSpace._parserLight)
        df_nondominated.to_csv('results_' + self.path + '_nondominated.csv', index=False)
        df_dominated.to_csv('results_' + self.path + '_dominated.csv', index=False)
        for el in np.unique(np.array(df_dominated['factors'])):
            df_dominated.loc[df_dominated['factors'] == el].to_csv(
                'results_' + self.path + '_dominated_factors=' + str(el) + '.csv', index=False)
        if self.model == 'UltraGCN':
            for el in np.unique(np.array(df_dominated['l'])):
                df_dominated.loc[df_dominated['l'] == el].to_csv(
                    'results_' + self.path + '_dominated_l=' + str(el) + '.csv', index=False)
        else:
            for el in np.unique(np.array(df_dominated['layers'])):
                df_dominated.loc[df_dominated['layers'] == el].to_csv(
                   'results_' + self.path + '_dominated_layers=' + str(el) + '.csv', index=False)
        for el in np.unique(np.array(df_nondominated['factors'])):
            df_nondominated.loc[df_nondominated['factors'] == el].to_csv(
                'results_' + self.path + '_nondominated_factors=' + str(el) + '.csv', index=False)
        if self.model == 'UltraGCN':
            for el in np.unique(np.array(df_nondominated['l'])):
                df_nondominated.loc[df_nondominated['l'] == el].to_csv(
                    'results_' + self.path + '_nondominated_l=' + str(el) + '.csv', index=False)
        else:
            for el in np.unique(np.array(df_nondominated['layers'])):
                df_nondominated.loc[df_nondominated['layers'] == el].to_csv(
                   'results_' + self.path + '_nondominated_layers=' + str(el) + '.csv', index=False)


if __name__ == '__main__':
    # set the path name at your convenience (notice that the an actual path name will likely be much more complex than the example one displayed below!)
    path = 'results/GCN.tsv'
    model_name = 'GCN' # default model
    if 'GCN' in path:
        model_name = 'GCN'
    elif 'LightGCN' in path:
        model_name = 'LightGCN'
    elif 'UltraGCN' in path:
        model_name = 'UltraGCN'
    else:
        raise NotImplementedError('Sorry, this graph recommender has not been implemented yet!')
    
    model = pd.read_csv(path, sep='\t')
    
    # comment or uncomment one of the two following lines of code at your convenience
    obj = ObjectivesSpace(model, {'nDCG': 'max', 'UserMADranking_WarmColdUsers': 'min'}, path, model_name)
    # obj = ObjectivesSpace(model, {'Recall': 'max', 'APLT': 'max'}, path, model_name)
    
    #print("-- DOMINATED --")
    #print(obj.get_dominated())
    #print("-- NON DOMINATED --")
    #print(obj.get_nondominated())
    #print("-- DISTANCES --")
    #print(obj.get_distances())
    print("-- STANDARD DEVIATION, MEAN --")
    print(obj.get_statistics())
    #obj.get_statistics_per_hyper()
    # obj.to_csv()
