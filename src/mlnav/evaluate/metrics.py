import datetime as dt
from os import path
from time import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from yellowbrick.cluster import SilhouetteVisualizer

from mlnav.utils.logger import Logger

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


class Unsupervised_Metrics:
    def __init__(self, model, data: pd.DataFrame) -> None:
        self.model = model  # Model to evaluate
        self.data = data  # Data to compute metrics
        self.labels = model.predict(data)  # Labels of data predicted by model
        self.metrics = None  # Dictionary where metrics are stored

    def get_metrics(
        self,
        compute_ch: bool,
        compute_dbs: bool,
        compute_sil: bool,
        sil_split: int = -1,
        sil_show: bool = False,
        sil_save: bool = False,
        save_path: str = "",
    ) -> Dict[str, np.float64]:
        """
        get_metrics() get all measures and save them

        :param compute_ch:          If True, compute ch
        :param compute_dbs:         If True, compute dbs
        :param compute_sil:         If True, compute silhouette
        :param sil_split:   Compute silhouette using sil_split instances, if sil_split=-1, use all
        :param dbcv_split:  Compute dbcv using dbcv_split instances, if dbcv_split=-1, use all
        :param sil_show:    If True, print silhouette plot
        :param sil_save:    If True, save silhouette plot
        :param save_path:   Path to save plots

        :return:            Dictionary metrics
        """

        st = time()
        Logger.getLogger().debug("Computing metrics...")

        ch = dbs = silhouette = dbcv = None

        if compute_ch:
            ch = self.calinski_harabasz()
        if compute_dbs:
            dbs = self.davies_bouldin_score()
        if compute_sil:
            silhouette = self.silhouette(sil_split, sil_show, sil_save, save_path)

        self.metrics = {
            "Silhouette": silhouette,
            "Calinski_harabasz": ch,
            "Davies_bouldin_score": dbs,
            "DBCV": dbcv,
        }

        for key, value in self.metrics.items():
            if value:
                self.metrics[key] = np.float64(value)

        et = time()
        Logger.getLogger().debug(
            f"All metrics computed in {str(dt.timedelta(seconds=(et-st)))}"
        )

        return {key: value for key, value in self.metrics.items() if value}

    def silhouette(
        self, silhouette_split, silhouette_show, silhouette_save, save_path
    ) -> np.float64:
        """
        silhouette() returns silhouette mean and plots silhouettes

        :param silhouette_split:   Compute silhouette using sil_split instances, if sil_split=-1, use all
        :param silhouette_show:    If True, print silhouette plot
        :param silhouette_save:    If True, save silhouette plot
        :param save_path:   Path to save plots

        return: silhouette score (float)
        """

        st_st = time()
        Logger.getLogger().debug("Computing Silhouette...")

        # Plot silhouette graph
        if silhouette_show == True:
            if silhouette_split == -1:
                reduced_data = self.data
            else:
                # Get labels of clusters
                labels = self.labels

                # Merge labels with attributes
                labels = labels.reshape(labels.shape[0], 1)
                df = pd.DataFrame(np.concatenate((labels, self.data), axis=1))
                cluster_split = (
                    silhouette_split // self.model.get_params(deep=False)["n_clusters"]
                )
                df = df.groupby(df.columns[0], group_keys=False).apply(
                    lambda x: x.sample(cluster_split)
                )

                # Convert to np array and delete indices column
                reduced_data = df.to_numpy()
                reduced_data = np.delete(reduced_data, 0, 1)

            # Visualizer
            silhouette_visualizer = SilhouetteVisualizer(
                self.model, is_fitted=True, colors="yellowbrick"
            )

            # Perform silhouette
            silhouette_visualizer.fit(reduced_data)

            # Save plot
            plots_path = path.join(save_path, "silhouette.png")

            silhouette_visualizer.show(outpath=plots_path if silhouette_save else "")

            score = silhouette_visualizer.silhouette_score_
        else:
            score = metrics.silhouette_score(
                self.data,
                self.labels,
                sample_size=(
                    silhouette_split if silhouette_split != -1 else len(self.data)
                ),
            )

        et_st = time()
        Logger.getLogger().info(
            f"Silhouette finished in {str(dt.timedelta(seconds=(et_st-st_st)))}"
        )

        return score

    def calinski_harabasz(self) -> np.float64:
        """
        calinski_harabasz() returns calinski harabasz score

        return: calinski harabasz score (float)
        """

        st_ch = time()
        Logger.getLogger().debug("Computing Calinski Harabasz...")

        # Compute calinski harabasz
        ch = metrics.calinski_harabasz_score(self.data, self.labels)

        et_ch = time()
        Logger.getLogger().info(
            f"Calinski Harabasz finished in {str(dt.timedelta(seconds=(et_ch-st_ch)))}"
        )

        return ch

    def davies_bouldin_score(self) -> np.float64:
        """
        davies_bouldin_score() returns davies bouldin score

        return: davies bouldin score (float)
        """

        st_db = time()
        Logger.getLogger().debug("Computing Davies Bouldin...")

        # Compute davies bouldin score
        db = metrics.davies_bouldin_score(self.data, self.labels)

        et_db = time()
        Logger.getLogger().info(
            f"Davies-Bouldin finished in {str(dt.timedelta(seconds=(et_db-st_db)))}"
        )

        return db
