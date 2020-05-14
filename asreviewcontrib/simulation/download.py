from urllib.request import urlretrieve
from pathlib import Path
import warnings
import os
import pickle
import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kstest
import scipy.stats as st

from asreview.models import get_model
from asreview.feature_extraction import get_feature_model
from asreview.balance_strategies import get_balance_model
from asreview import ASReviewData
from scipy.optimize import minimize

from asreviewcontrib.simulation.exponential_tail import ExpTailNorm


DISTRIBUTIONS = [
     st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,  #noqa
     st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,  #noqa
     st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,  #noqa
     st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,  #noqa
     st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,  #noqa
     st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,  #noqa
     st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,  #noqa
     st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,  #noqa
     st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,  #noqa
     st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy  #noqa
 ]

SLOW_DISTRIBUTIONS = [
    st.levy_stable
]

base_url = "https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/"  #noqa

files = {
    "wilson.csv": "Appenzeller-Herzog_Wilson/output/output_csv_wilson.csv",
    "bannach.csv": "Bannach-Brown%20et%20al.%20(2019)/output/Bannach-Brown_2019.csv",  # noqa
    "ace.csv": "Cohen_EBM/output/ACEInhibitors.csv",
    "statins.csv": "Cohen_EBM/output/Statins.csv",
    "ptsd.csv": "Van_de_Schoot_PTSD/output/PTSD_VandeSchoot_18.csv",
    "hall.csv": "Four%20Software%20Engineer%20Data%20Sets/output/Software_Engineering_Hall.csv",  #noqa
    "wahono.csv": "Four%20Software%20Engineer%20Data%20Sets/output/Software_Engineering_Wahono.csv",  #noqa
    "radjenovic.csv": "Four%20Software%20Engineer%20Data%20Sets/output/Software_Engineering_Radjenovic.csv",  #noqa
    "kitchenham.csv": "Four%20Software%20Engineer%20Data%20Sets/output/Software_Engineering_Kitchenham.csv",  #noqa
    "vandis.csv": "van_Dis%20et%20al.%20(2020)/output/van_Dis_2020.csv"
}


def optimize_distribution(cache_fp, optimization_fp, data_dir="data",
                          compute_dist=False):
    model = get_model("nb")
    feature_model = get_feature_model("tfidf")
    balance_model = get_balance_model("double")

    try:
        with open(cache_fp, "rb") as f:
            all_results = pickle.load(f)
    except FileNotFoundError:
        all_results = {}

    for file_name, part_url in files.items():
        url = base_url+part_url
        compute_decision_function(file_name, data_dir, url, model,
                                  feature_model, balance_model, all_results)
        if compute_dist:
            compute_distributions(file_name, all_results)

    optimize_power_tail(all_results, optimization_fp)
    with open(cache_fp, "wb") as f:
        pickle.dump(all_results, f)
    return


def plot_distributions(all_results):
    global_sorted_dist = get_ordered_distributions(all_results)

    dist_result = global_sorted_dist[list(global_sorted_dist)[0]]
    dist = dist_result["dist"]
    all_df_one = []
    for file_name, res in dist_result["results"].items():
        df_one = all_results[file_name]["df_one"]
        df_zero = all_results[file_name]["df_zero"]
        x = np.linspace(np.min(np.append(df_one, df_zero)),
                        np.max(np.append(df_one, df_zero)), 50)
        param = res["param"]
        print(param)
        one_dist = dist(loc=param[-2], scale=param[-1], *param[:-2])
        plt.plot(x, one_dist.pdf(x))
        all_df_one.append(df_one)
    plt.hist(all_df_one, 30, histtype="bar", density=True, label=list(files))
    plt.title(dist.name)
    plt.legend()
    plt.show()


def compute_decision_function(file_name, base_dir, url, model, feature_model,
                              balance_model, result):
    if file_name not in result:
        result[file_name] = {}
    if "df_one" in result[file_name] and "df_zero" in result[file_name]:
        return

    if not Path(base_dir).is_dir():
        os.makedirs(base_dir)

    data_fp = Path(base_dir, file_name)
    if not data_fp.is_file():
        urlretrieve(url, data_fp)

    as_data = ASReviewData.from_file(data_fp)
    X = feature_model.fit_transform(as_data.texts, as_data.headings,
                                    as_data.bodies, as_data.keywords)

    y = as_data.labels
    one_idx = np.where(y == 1)[0]
    zero_idx = np.where(y == 0)[0]
    all_idx = np.arange(len(as_data))

    one_proba = []
    zero_proba = []
    for _ in range(10):
        for cur_one_idx in one_idx:
            cur_zero_idx = np.random.choice(zero_idx, len(zero_idx)//2,
                                            replace=False)
            train_idx = np.delete(
                all_idx, np.concatenate(([cur_one_idx], cur_zero_idx)))
            X_train, y_train = balance_model.sample(X, y, train_idx, {})
            model.fit(X_train, y_train)
            correct_one_proba = model.predict_proba(X[cur_one_idx])[0, 1]
            correct_zero_proba = model.predict_proba(X[cur_zero_idx])[:, 1]
            one_proba.append(correct_one_proba)
            zero_proba.extend(correct_zero_proba)

    df_one = -np.log(1/np.array(one_proba)-1)
    df_zero = -np.log(1/np.array(zero_proba)-1)

    result[file_name]["df_one"] = df_one
    result[file_name]["df_zero"] = df_zero
    return


def optimize_power_tail(all_results, output_file):
    def max_likelihood(param):
        npar = len(param) - 2*len(all_results)
        extra_param = param[:npar]
        all_mu = param[npar:len(all_results)+npar]
        all_sigma = param[npar+len(all_results):]

        log_likelihood = 0
        for i, results in enumerate(all_results.values()):
            df_one = results["df_one"]
            mu = all_mu[i]
            sigma = all_sigma[i]
            dist = ExpTailNorm(mu, sigma, *extra_param)
            p_val = dist.pdf(df_one)
            log_likelihood += np.sum(np.log(p_val))

        return -log_likelihood

    all_mu_range = []
    all_sigma_range = []
    all_mu_start = []
    all_sigma_start = []
    for i, results in enumerate(all_results.values()):
        df_one = results["df_one"]
        mu_range = (np.min(df_one), np.max(df_one))
        est_sigma = np.sqrt(np.var(df_one))
        sigma_range = (0.7*est_sigma, 1.3*est_sigma)
        all_mu_range.append(mu_range)
        all_sigma_range.append(sigma_range)
        all_mu_start.append(np.average(df_one))
        all_sigma_start.append(est_sigma)

    common_range = [
        (1, 2),
    ]
    common_start = [1.5]
    bounds = common_range+all_mu_range+all_sigma_range
    x0 = np.array(common_start+all_mu_start+all_sigma_start)

    minim_result = minimize(fun=max_likelihood, x0=x0, bounds=bounds)
    x_opt = minim_result.x
    print(minim_result)
    df_one_norm = []
    min_df = []
    npar = len(common_range)
    for i, results in enumerate(all_results.values()):
        mu = x_opt[npar+i]
        sigma = x_opt[npar+len(all_results)+i]
        df_one = results["df_one"]
        df_one_norm.extend((df_one-mu)/sigma)
        min_df.append(np.min((df_one-mu)/sigma))

    min_df = np.sort(min_df)
    min_df_cum = (np.arange(len(min_df))+1)/len(min_df)
    plt.plot(min_df, min_df_cum)
    plt.show()
    param = [0, 1] + x_opt[:npar].tolist()
    plot_one_dist(df_one_norm, param)

    sorted_df = np.sort(df_one_norm)
    df_cum = (np.arange(len(sorted_df))+1)/len(sorted_df)
    plt.plot(sorted_df, df_cum)
    plt.show()

    opt_results = {
        "min_df": (min_df.tolist(), min_df_cum.tolist()),
        "extra_param": param[2:],
        "cum_df": (sorted_df.tolist(), df_cum.tolist()),
    }
    with open(output_file, "w") as f:
        json.dump(opt_results, f)


def self_check(dist, x_max=20, n_sample=100000):
    x_range = np.linspace(-x_max, x_max, n_sample)
    y = dist.pdf(x_range)
    return np.sum(y)/(n_sample/(2*x_max))


def plot_one_dist(df_one, param):
    dist_exp = ExpTailNorm(*param)
    x_range = np.linspace(-6, 6)
    plt.hist(df_one, 40, density=True)
    plt.plot(x_range, st.norm.pdf(x_range, loc=param[0], scale=param[1]),
             label="Normal tail")
    plt.plot(x_range, dist_exp.pdf(x_range), label="Exponential tail")
    plt.legend()
    plt.show()
    plt.plot(x_range, dist_exp.cdf(x_range), label="Exponential tail")


def compute_distributions(file_name, result):
    file_result = result[file_name]

    if "sorted_dist" in file_result:
        return

    df_one = file_result["df_one"]
    df_zero = file_result["df_zero"]

    all_dist = []
    for dist in DISTRIBUTIONS:
        with warnings.catch_warnings():
            if dist in SLOW_DISTRIBUTIONS:
                continue
            warnings.filterwarnings('ignore')
            param = dist.fit(df_one)
            if len(param) > 3:
                continue
            ks_stat, ks_p = kstest(df_one, dist.cdf, param)
            if np.isnan(ks_p):
                continue
            all_dist.append({"dist": dist, "p": ks_p, "param": param})

    all_dist = sorted(all_dist, key=lambda x: -x["p"]/len(x["param"]))

    file_result["sorted_dist"] = all_dist


def get_ordered_distributions(result):
    global_dist = {}
    for file_name in result:
        sorted_dist = result[file_name]["sorted_dist"]
        for rank, dist_item in enumerate(sorted_dist):
            dist_name = dist_item["dist"].name
            if dist_name not in global_dist:
                global_dist[dist_name] = {
                    "dist": dist_item["dist"],
                    "results": {
                    }
                }
            res = global_dist[dist_name]["results"]
            res[file_name] = {
                "param": dist_item["param"],
                "rank": rank,
            }

    def get_total_rank(x):
        results = global_dist[x]["results"]
        ranks = [x["rank"] for x in results.values()]
        return np.sum(ranks) + 4*(len(result)-len(ranks))*len(DISTRIBUTIONS)

    global_sorted_dist = {k: global_dist[k]
                          for k in sorted(global_dist, key=get_total_rank)}
    return global_sorted_dist
