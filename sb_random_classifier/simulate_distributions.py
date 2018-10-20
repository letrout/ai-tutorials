#!/usr/bin/env python
"""
Simulate the test result distributions we see from repeated runs of the
sysbench memory benchmark. Qualitative classifications of these distributions:
1. "clean": Normal/Gaussian distribution with <1% stddev/mean
2. "fuzz": Normal/Gaussian distribution with >5% stddev/mean
3. "peaks": "clean" with occasional peaks of x * mean (x>1), occasional <<50%
4. "dips": "clean" with occasional drops of y * mean (0<y<1), occasional <<50%
5(?) "bimodal": "clean with occasional peaks/dips, where occasional is ~50%

Intended for use in training a xNN model to classify sysbench mem results
by their amount and type of randomness

Observed:
clean: cv of <0.5%
fuzz: cv up to 40%
peaks of 10-40% freq, 1.5-1.6x baseline
dips of 5-10% freq, 0.5-0.75x baseline
We should generate a training dataset that goes somewhat outside these
observations to avoid overfitting

In this script I use "cv" as shorthand for coefficient of variation,
or stddev/mean
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# TODO: make these script args
DEFAULT_SAMPLES = 96
DEFAULT_SERIES_PER_LABEL = 10
DEFAULT_MEAN = 100

# TODO: these will be deprecated?
CLEAN_CVS = [0.001, 0.003, 0.005, 0.01, 0.02]
FUZZ_CVS = [0.10, 0.20, 0.25, 0.30, 0.40]
PEAK_BASE_PARAMS = {
    'mean': 100,
    'cv': 0.02
}

class RandomDataset:
    def __init__(self):
        self.__df_clean = None
        self.__df_fuzz = None
        self.__df_peaks = None
        self.__df_dips = None
        self.__df_bimodal = None

        self.__df_clean = create_normal_df(
            num_series=DEFAULT_SERIES_PER_LABEL,
            num_samples=DEFAULT_SAMPLES,
            mean=DEFAULT_MEAN,
            cvs=CLEAN_CVS)
        self.__df_fuzz = create_normal_df(
            num_series=DEFAULT_SERIES_PER_LABEL,
            num_samples=DEFAULT_SAMPLES,
            mean=DEFAULT_MEAN,
            cvs=FUZZ_CVS)

    @property
    def df_clean(self):
        return self.__df_clean

    @property
    def df_fuzz(self):
        return self.__df_fuzz


def create_normal_series(cv, mean=DEFAULT_MEAN, count=DEFAULT_SAMPLES):
    """
    Create a series of normal distribution
    :param mean: mean of the distribution
    :param cv: coefficient of variation of the distribution
    :param count: length of the data series created
    :return: Series with the specified distribution
    """
    return pd.Series(np.random.normal(mean, cv * mean, count))

def create_normal_df(
        num_series=DEFAULT_SERIES_PER_LABEL,
        num_samples=DEFAULT_SAMPLES,
        mean=DEFAULT_MEAN,
        cvs=[]):
    df = pd.DataFrame()
    i = 0
    j = -1
    while (i < num_series):
        for cv in cvs:
            i += 1
            j += 1
            series = create_normal_series(cv, mean, num_samples)
            df.insert(j, j, series)
    return df

def create_peak_series(peak_mean, peak_cv, peak_freq, count, series_base=None):
    """
    Create a 0-series with occasional peaks (or dips)
    :param peak_mean: mean height of peaks (negative number for dips)
    :param peak_cv: CV of the height of the peaks/dips
    :param peak_freq: frequency (0-1) of the peaks/dips
    :param count: total length of the series
    :return: 0-series with occasional peaks or dips
    """
    # Create a "base" series on which we will apply our peaks
    if series_base is None or count != len(series_base.index):
        series_base = create_normal_series(
            CLEAN_CVS[0], DEFAULT_MEAN, count
        )
    peaks = lambda x: np.random.normal(peak_mean, peak_cv * peak_mean, 1)[0]\
       if np.random.uniform() < peak_freq else 0
    return series_base.apply(peaks)

def create_peak_df(
        repeat=5, num_rows=DEFAULT_SAMPLES, peak_means=[], peak_cvs=[], peak_freqs=[]):
    df = pd.DataFrame()
    j = -1
    for i in range(0, repeat, 1):
        for mean in peak_means:
            for cv in peak_cvs:
                for freq in peak_freqs:
                    j += 1
                    series = create_peak_series(mean, cv, freq, num_rows)
                    df.insert(j, j, series)
    return df

def plot_series_hist(df, bins=20):
    df.plot.hist(bins=bins)
    plt.show()

def plot_series_line(df):
    df.plot.line()
    plt.show()

if __name__ == '__main__':
    np.random.seed()
    ranset = RandomDataset()
    print(ranset.df_clean)
    print(ranset.df_fuzz)
    ranset.df_clean.shape
    #peak_series = create_peak_series(40, 0.05, 0.1, 100)
    #plot_series_hist(peak_series)
    #plot_series_line(peak_series)
