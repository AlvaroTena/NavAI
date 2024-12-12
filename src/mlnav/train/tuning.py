from datetime import timedelta
from time import time
from typing import Dict, List, Union

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from mlnav.utils.logger import Logger, tqdm_progress


def params_eval(results: List[Dict], best_trial: Union[Dict, None]) -> Dict:
    """Evaluate Trials results from Bayesian Optimization

    Args:
        results : List containing dictionaries results from optimization
            The results where search best global score.
        best_trials : Dictionary with best trial
            Best trial based on loss from optimization.

    Returns:
        params : Dictionary
            The best combination of params with best global score.
    """
    best_trial_result = best_trial["result"]

    sil_scores = [result["silhouette"] for result in results]
    davies_scores = [result["davies_bouldin"] for result in results]
    calinski_scores = [result["calinski"] for result in results]

    best_trial_sil_score = (
        (best_trial_result["silhouette"] - -1) / (1 - -1) if best_trial else 0
    )
    best_trial_davies_score = (best_trial_result["davies_bouldin"] - 0) / (
        max(davies_scores) - 0 if best_trial else 0
    )
    best_trial_calinski_score = (best_trial_result["calinski"] - 0) / (
        max(calinski_scores) - 0 if best_trial else 0
    )
    best_trial_score = (
        best_trial_sil_score + best_trial_calinski_score + (1 - best_trial_davies_score)
    )

    sil_scores = [(sil - -1) / (1 - -1) for sil in sil_scores]
    davies_scores = [
        (davies - 0) / (max(davies_scores) - 0) for davies in davies_scores
    ]
    calinski_scores = [
        (calinski - 0) / (max(calinski_scores) - 0) for calinski in calinski_scores
    ]
    scores = [
        sil + calinski + (1 - davies)
        for sil, davies, calinski in zip(sil_scores, davies_scores, calinski_scores)
    ]

    if max(scores) != best_trial_score:
        Logger.getLogger().debug(
            f"Bayesian optimization best loss {best_trial_score} differs from best mixed score {max(scores)}."
        )
    return results[np.argmax(scores)]["params"]


class Model_Tuning:
    def __init__(
        self,
        model,
        param_grid: dict,
        verbose: bool = True,
        random_state: int = 42,
    ) -> None:
        self.model = model  # model to tune
        self.param_grid = param_grid
        self.verbose = verbose
        self.random_state = random_state

    def bayesianOptimizer(self, data, iters, trials_save_file=""):
        def objective(params):
            model = type(self.model)(**params).fit(data)

            sil_score = silhouette_score(
                data, model.labels_, sample_size=5000, random_state=self.random_state
            )
            davies_score = davies_bouldin_score(data, model.labels_)
            calinski_score = calinski_harabasz_score(data, model.labels_)

            return {
                "loss": davies_score,
                "silhouette": sil_score,
                "davies_bouldin": davies_score,
                "calinski": calinski_score,
                "params": params,
                "status": STATUS_OK,
            }

        st = time()
        bayes_trials = Trials()
        rstate = np.random.default_rng(self.random_state)
        fmin(
            fn=objective,
            space=self.param_grid,
            algo=tpe.suggest,
            max_evals=iters,
            trials=bayes_trials,
            rstate=rstate,
            show_progressbar=tqdm_progress,
            verbose=self.verbose,
            trials_save_file=trials_save_file,
        )
        Logger.getLogger().debug(
            f"Bayesian optimization finished in {str(timedelta(seconds=time()-st))}"
        )

        # return params_eval(bayes_trials.results, bayes_trials.best_trial)
        return bayes_trials.best_trial["result"]["params"]
