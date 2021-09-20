import random
import warnings
from collections import Counter
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
from matplotlib.collections import LineCollection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

random.seed(0)
np.random.seed(0)


# sklearnのTransformerにしたので、fit_tramsformを呼ぶと外れ値の閾値を返す
class OutlyerModel(BaseEstimator, TransformerMixin):
    """GMMを使って外れ値を計算するクラス

    sikit-learnのTransfomerMixinを実装しているため、fit_transfomを呼び出すことで、外れ値の範囲を計算する.

    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]
    """

    def __init__(self, search_list: list[int] = list(range(1, 70)), warning_filter=True):
        """! GMMを使って外れ値を計算するクラス

        Args:
            search_list ([list[int]], optional): GMMのコンポーネント数を探索する範囲. Defaults to list(range(1, 70)).
            warning_filter (bool, optional): GMMのワーニングを抑制するフラグ. Defaults to True.
        """
        self._search_list = search_list
        self._warning_filter = warning_filter

    # 各インデックスのクラス数のBICを計算、もしくはキャッシュから得る

    def get_bic(self, index, x):
        """! 指定されたコンポーネントのインデックスのbicをけいさんする

        Args:
            index ([type]): [description]
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        bics = self.bics_
        gmms = self.gmms_
        if (bics[index] is None):
            if self._warning_filter:
                warnings.filterwarnings('ignore')
            gmm = GaussianMixture(
                n_components=self._search_list[index], reg_covar=1.0e-3)
            gmm = gmm.fit(x)
            bic = gmm.bic(x)
            gmms[index] = gmm
            bics[index] = bic
            self.gmm_fit_count_ += 1
            return bic
        return bics[index]

    def gmm_search(self, x):
        # 最小のBICのクラス数を三分探索をする
        # https://qiita.com/DaikiSuyama/items/84df26daad11cf7da453
        sarch_list_size = len(self._search_list)
        left = 0
        right = sarch_list_size - 1

        self.gmms_ = [None] * sarch_list_size
        self.bics_ = [None] * sarch_list_size

        self.gmm_fit_count_ = 0

        # right=left,left+1,left+2のいずれかになれば探索終了
        while(left + 2 < right):
            centor1 = left + (right - left) // 3
            centor2 = right - (right - left) // 3

            # 3分点を計算する
            centor1_bic = self.get_bic(centor1, x)
            centor2_bic = self.get_bic(centor2, x)
            if centor1_bic < centor2_bic:
                right = centor2
            else:
                left = centor1

        # left-rightの間の最小値が最小値
        min_bic = float('inf')
        min_index = -1
        for i in range(left, right+1):
            bic = self.get_bic(i, x)
            if bic < min_bic:
                min_bic = bic
                min_index = i
        self.min_index_ = min_index
        self.gmm_ = self.gmms_[min_index]
        self.n_components_ = self._search_list[min_index]
        return

    def fit(self, X):
        return self

    def predict(self, X):
        x = X.reshape((-1, 1))
        return self.gmm_.predict(x)

    def transform(self, X):
        # 最良のbicが得られるgmmを探索する
        self.size_ = len(X)
        x_reshaped = X.reshape((-1, 1))
        self.x_ = X
        self.gmm_search(x_reshaped)

        # σプロットから閾値を計算する
        # σプロットの3-5σの領域から推定
        # 3-5σの領域が離散的だったら、5σがあれば、5σ、なければ最大、最小を使う
        hist, bin_edges = np.histogram(X, 200)
        x = np.mean(np.stack([bin_edges[:-1], bin_edges[1:]], axis=1), axis=1)
        cumsum_hists = np.cumsum(hist)
        sigmas = self.calc_sigma(cumsum_hists, np.full(
            len(cumsum_hists), cumsum_hists[-1]))
        self.x_org_ = x
        self.sigmas_org_ = sigmas

        _, bin_edges = np.histogram(self.x_, 200)
        #x = np.mean(np.stack([bin_edges[:-1], bin_edges[1:]], axis=1), axis=1)
        expand = (bin_edges[-1] - bin_edges[0])
        x = np.linspace(bin_edges[0] - expand,
                        bin_edges[-1] + expand, 200 * 100)
        gmm_pdf = self.GMMPDF(self.gmm_, x.shape[0])
        counts = gmm_pdf.pdf(x)
        cumsum_counts = np.cumsum(counts)
        sigmas = self.calc_sigma(cumsum_counts, np.full(
            len(cumsum_counts), cumsum_counts[-1]))
        select = (-5 < sigmas) & (sigmas < 5)
        x = x[select]
        sigmas = sigmas[select]
        self.x_sigmas_ = x
        self.sigmas_ = sigmas

        thr_low = x[0]
        thr_high = x[-1]

        # 3-5σ領域の線形近似
        # 左
        select = (-5 < sigmas) & (sigmas < -3)
        x_left = x[select]
        sigmas_left = sigmas[select]
        model_left = LinearRegression()
        model_left.fit(x_left.reshape((-1, 1)), sigmas_left.reshape((-1, 1)))
        thr_low2 = thr_low
        if model_left.coef_ > 0:
            thr_low2 = (-5 - model_left.intercept_)/model_left.coef_
            start_low2 = (-3 - model_left.intercept_)/model_left.coef_
        self.thr_low2_ = thr_low2
        self.start_low2_ = start_low2
        # 右
        select = (3 < sigmas) & (sigmas < 5)
        x_right = x[select]
        sigmas_right = sigmas[select]
        model_right = LinearRegression()
        model_right.fit(x_right.reshape((-1, 1)),
                        sigmas_right.reshape((-1, 1)))
        thr_high = thr_high
        if model_right.coef_ > 0:
            thr_high2 = (5 - model_right.intercept_)/model_right.coef_
            start_high2 = (3 - model_right.intercept_)/model_right.coef_
        self.thr_high2_ = thr_high2
        self.start_high2_ = start_high2

        self.thr_low_ = thr_low
        self.thr_high_ = thr_high
        reject_count = len(X[(X < thr_low) | (X > thr_high)])
        self.reject_count_ = reject_count
        self.reject_rate_ = reject_count / float(self.size_)

        return thr_low, thr_high, self.reject_rate_

    def plot(self, search_ax=None, hist_ax=None, prob_ax=None):
        self.plot_search_component(search_ax)
        self.plot_hist(hist_ax)
        self.plot_prob(prob_ax)
        ax = plt.figure().add_subplot()
        self.sigma_plot(self.x_, ax=ax)
        ax.plot(self.x_org_, self.sigmas_org_)
        ax.plot(self.x_sigmas_, self.sigmas_)
        ax.axhline(-3, xmin=0, xmax=1, linestyle="--")
        ax.axhline(3, xmin=0, xmax=1, linestyle="--")
        ax.axhline(-5, xmin=0, xmax=1, linestyle="--")
        ax.axhline(5, xmin=0, xmax=1, linestyle="--")

        ax.axvline(self.thr_low_, ymin=0, ymax=1, color="red", linestyle='--')
        ax.axvline(self.thr_high_, ymin=0, ymax=1, color="red", linestyle='--')
        ax.axvline(self.thr_low2_, ymin=0, ymax=1,
                   color="blue", linestyle='--')
        ax.axvline(self.thr_high2_, ymin=0, ymax=1,
                   color="blue", linestyle='--')

        lines = [[(self.thr_low2_, -5), (self.start_low2_, -3)],
                 [(self.start_high2_, 3), (self.thr_high2_, 5)]]
        lc = LineCollection(lines, colors=['green', 'green'])
        ax.add_collection(lc)

    def plot_search_component(self, ax=None, recalc_full_bic=False):
        if ax is None:
            ax = plt.figure().add_subplot()

        n_search = sum(x is not None for x in self.bics_)
        if recalc_full_bic:
            for i in range(len(self._search_list)):
                self.get_bic(i, self.x_)
        ax.plot(self._search_list, self.bics_)
        s = f"components: {self.n_components_}\nsearch: {n_search}"
        ax.text(0.99, 0.99, s, va='top', ha='right', transform=ax.transAxes)

    def plot_hist(self, ax=None):
        if ax is None:
            ax = plt.figure().add_subplot()

        bin = ceil(sqrt(self.size_))
        ax.hist(self.x_, bins=bin, log=True)
        ax.axvline(self.thr_low_, ymin=0, ymax=1, color="red", linestyle='--')
        ax.axvline(self.thr_high_, ymin=0, ymax=1, color="red", linestyle='--')
        ax.text(self.thr_low_, ax.axis()[3], str(
            self.thr_low_), va="top", ha="left")
        ax.text(self.thr_high_, ax.axis()[3], str(
            self.thr_high_), va="top", ha="right")
        ax.text(1.02, 0.5, f"reject_rate:{self.reject_rate_:.3e}",
                va="center", transform=ax.transAxes, rotation=270)

    def plot_prob(self, ax=None):
        if ax is None:
            ax = plt.figure().add_subplot()

        # 分布のラインを描画
        bin = ceil(sqrt(self.size_))
        ax.hist(self.x_, bins=bin)
        x = np.linspace(np.min(self.x_), np.max(
            self.x_), bin if bin >= 100 else 100)
        gmm = self.gmm_
        y_sum = np.zeros(x.shape[0])
        ys = np.zeros((self.n_components_, x.shape[0]))
        for i in range(self.n_components_):
            gd = scipy.stats.norm.pdf(
                x, gmm.means_[i, -1], np.sqrt(gmm.covariances_[i]))
            y = gmm.weights_[i] * gd
            y = y[0]
            ys[i] = y
            y_sum += y
        magnify = self.size_ / np.sum(y_sum)
        y_sum *= magnify
        ys *= magnify
        for i in range(self.n_components_):
            ax.plot(x, ys[i], label=f'components: {i}', color="red", alpha=0.5)
        ax.plot(x, y_sum, label=f'components: {i}', color="red")

    def calc_sigma(self, x, all):
        return stats.norm.ppf((x-0.3)/(all+0.4))

    def sigma_plot_from_hist(self, counts, bins, bin_is_edge=True, ax=None):
        if ax is None:
            ax = plt.figure().add_subplot()

        if bin_is_edge:
            x = np.mean(np.stack([bins[:-1], bins[1:]], axis=1), axis=1)
        else:
            x = bins

        cumsum_counts = np.cumsum(counts)
        sigmas = self.calc_sigma(cumsum_counts, np.full(
            len(cumsum_counts), cumsum_counts[-1]))

        ax.plot(x, sigmas)

    def sigma_plot(self, x, bin_num=200, ax=None):
        counts, bins = np.histogram(x, bin_num)
        self.sigma_plot_from_hist(counts, bins, ax)

    def sigma_plot_from_pdf(self, f, bins, ax=None):
        x = np.mean(np.stack([bins[:-1], bins[1:]], axis=1), axis=1)
        counts = f(x)
        self.sigma_plot_from_hist(counts, bins, ax=ax)

    class GMMPDF:
        def __init__(self, gmm, n_size):
            self.gmm = gmm
            self.n_size = n_size

        def pdf(self, x):
            gmm = self.gmm
            y = np.zeros(x.shape[0])
            for i in range(gmm.n_components):
                y += stats.norm.pdf(x, gmm.means_.reshape(-1)[i], np.sqrt(
                    gmm.covariances_.reshape(-1)[i]))*gmm.weights_[i]
            magnify = self.n_size / np.sum(y)
            return y * magnify
