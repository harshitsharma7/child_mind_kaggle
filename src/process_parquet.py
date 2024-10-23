import pandas as pd 
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

class ParquetDataset():
    def __init__(self, path_to_dir):
        self.path_to_dir = path_to_dir
        self.feature_dict = {"mav":(self.mav, ["X","Y","Z", "enmo", "light"]), 
                             "ssc":(self.ssc, ["X","Y","Z", "enmo", "light"]),
                             "var":(self.var, ["X","Y","Z", "enmo", "light"])}

    
    def process_parquet(self, filename, dirname):
        _id = filename.split('=')[1]
        df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
        df.drop('step', axis=1, inplace=True)
        features = np.empty((0,))
        for f in self.feature_dict.values():
            operation, operands = f[0], f[1]
            x = df[operands].values
            features = np.concatenate((features, operation(x).reshape(-1)))
        return features, _id

    def generate_dataset(self, save_csv=True) -> pd.DataFrame:
        ids = os.listdir(self.path_to_dir)
        
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda fname: self.process_parquet(fname, self.path_to_dir), ids), total=len(ids)))
        
        stats, indexes = zip(*results)
        columns = [f"{key}_{value}" for key, (_, values) in self.feature_dict.items() for value in values]

        df = pd.DataFrame(stats, columns=columns)
        df['id'] = indexes
        return df
    

    """
    features
    ---------------------------------------------------------
    """

    def mav(self, x):                                      # mean absolute value
        return sum(abs(x)) / x.shape[0]

    def ssc(self, x, delta = 1):                           # slope of sign change
        f = lambda x: (x >= delta).astype(float)
        return sum(f(-(x[1:-1, :] - x[:-2, :])*(x[1:-1] - x[2:])))
    
    def var(self, x):                                      # variance
        return sum((x-np.mean(x, axis=0))** 2 ) / (x.shape[0] - 1)

    #TODO sunlight exposure (proxy for outdoor time)

    #TODO day vs night ambient light levels

    #TODO does anglez remains the same for long periods of time

    #TODO weekday vs weekend light exposure

    #TODO weekday vs weekend physical activity


