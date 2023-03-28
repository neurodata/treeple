from sktree.ensemble import ObliqueRandomForestClassifier

from ._common import Benchmark, Estimator, Predictor
from ._datasets import (
    _20newsgroups_highdim_dataset,
    _20newsgroups_lowdim_dataset,
    _synth_classification_dataset,
)
from ._utils import make_gen_classif_scorers


class ObliqueRandomForestClassifierBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for RandomForestClassifier.
    """

    param_names = ["representation", "n_jobs"]
    params = (["dense"], Benchmark.n_jobs_vals)

    def setup_cache(self):
        super().setup_cache()

    def make_data(self, params):
        representation, n_jobs = params

        data = _20newsgroups_lowdim_dataset()

        return data

    def make_estimator(self, params):
        representation, n_jobs = params

        n_estimators = 500 if Benchmark.data_size == "large" else 100

        estimator = ObliqueRandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=10,
            max_features="log2",
            n_jobs=n_jobs,
            random_state=0,
        )

        return estimator

    def make_scorers(self):
        make_gen_classif_scorers(self)
