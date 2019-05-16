'''
Analysis and reading of log files.

Merged versions of functions work on the results of all files at the same time.
'''

import os
import re
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc


def read_labels(pickle_file):
    """ Load the labels from the pickle file. """
    with open(pickle_file, "rb") as fp:
        _, labels, _, _ = pickle.load(fp)
    return labels


def read_json_results(data_dir):
    """
    Find all results in a directory and read them in memory.
    Assume that all files in this directory have the same model parameters.

    Arguments
    ---------
    data_dir: str
        Directory in which to find any log files.

    Returns
    -------
    dict:
        Dictionary containing the results.
    """
    json_data = {}
    files = os.listdir(data_dir)
    if not files:
        print(f"Error: {data_dir} is empty")
        return None

    min_queries = int(10**9)
    for json_file in files:
        if not re.match(r'^results', json_file):
            continue
        with open(os.path.join(data_dir, json_file), "r") as fp:
            json_data[json_file] = json.load(fp)

        i = 0
        while i < len(json_data[json_file]):
            if str(i) not in json_data[json_file]:
#                 print(f"i = {i}")
#                 print(f"{json_data[json_file].keys()}")
                min_queries = min(min_queries, i)
                break
            i += 1

    # Make sure they all have the same number of queries.
#     print(f"min_queries: {min_queries}")
    for json_file in files:
        i = min_queries
        max_i = len(json_data[json_file])
        while i < max_i:
            if str(i) not in json_data[json_file]:
                break
            del json_data[json_file][str(i)]
#             print(f"Warning: not using query {i} from file {json_file}")
            i += 1
#         print(f"{json_data[json_file].keys()}")

    return json_data


def reorder_results(old_results):
    """
    From a dictionary of results, create a better ordered result.
    The hierarchy of the new dictionary is:
    logname -> query_id -> filename -> data.

    Arguments
    ---------
    old_results: dict
        Results to reorder.

    Returns
    dict:
        Reordered results.
    """
    results = {}
    first_file = list(old_results.keys())[0]
    lognames = old_results[first_file]["0"].keys()
    queries = old_results[first_file].keys()
    files = old_results.keys()

    for log in lognames:
        results[log] = {}
        for query in queries:
            try:
                int(query)
            except ValueError:
                continue
            results[log][query] = []
            for fp in files:
                results[log][query].append(old_results[fp][query][log])
    return results


def get_num_queries(results):
    """ Get the number of queries from the non-reordered results. """
    num_queries = []
    for filename in results:
        cur_num = []
        for query in results[filename]:
            # All results are an integer number.
            try:
                int(query)
            except ValueError:
                continue
            # Count the number of labeled samples each query.
            d_num = len(results[filename][query]["labelled"])
            if len(cur_num) == 0:
                cur_num.append(d_num)
            else:
                cur_num.append(d_num + cur_num[-1])
        # Assert that the number of queries is the same for all files.
        if len(num_queries) == 0:
            num_queries = cur_num
        else:
            assert num_queries == cur_num
    return num_queries


def _split_probabilities(proba, labels):
    """ Split probabilities into the two classes for further processing. """
    class_proba = [[], []]
    for res in proba:
        sample_id = res[0]
        prob = res[1]
        true_cat = labels[sample_id]
        class_proba[true_cat].append(prob)
    for i in range(2):
        class_proba[i] = np.array(class_proba[i])
    return class_proba


def _speedup(proba, labels, n_allow_miss=0, normalize=True):
    """
    Compute the number of non included papers below the worst (proba)
    one that is included in the final review.

    Arguments
    ---------
    proba: float
        List of probabilities with indices.
    labels: int
        True categories of each index.
    n_allow_miss: int
        Number of allowed misses.
    normalize: bool
        Whether to normalize with the expected number.

    Returns
    -------
    float:
        Number of papers that don't need inclusion.
    """
    class_proba = _split_probabilities(proba, labels)
    for i in range(2):
        class_proba[i].sort()

    # Number of 0's / 1's / total in proba set.
    num_c0 = class_proba[0].shape[0]
    num_c1 = class_proba[1].shape[0]
    num_tot = num_c0 + num_c1

    # Determine the normalization factor, probably not quite correct.
    if normalize and num_c1 > 0:
        norm_fact = (1+2*n_allow_miss)*num_tot/(2*num_c1)
    else:
        norm_fact = 1

    # If the number of 1 labels is smaller than the number allowed -> all.
    if num_c1 <= n_allow_miss:
        return num_tot/norm_fact
    lowest_prob = class_proba[1][n_allow_miss]
    for i, val in enumerate(class_proba[0]):
        if val >= lowest_prob:
            return (i+n_allow_miss)/norm_fact
    return num_c0/norm_fact


def _avg_false_neg(proba, labels):
    res = np.zeros(len(proba))
    proba.sort(key=lambda x: x[1])
#     print(proba[:100])
    n_one = 0
    for i, item in enumerate(proba):
        sample_id = item[0]
        true_cat = labels[sample_id]
        if true_cat == 1:
            n_one += 1
        res[i] = n_one
    return res


def _limits_merged(proba, labels, p_allow_miss):
    n_samples = len(proba[0])
    n_proba = len(proba)
    false_neg = np.zeros(n_samples)
    for sub_proba in proba:
        false_neg += _avg_false_neg(sub_proba, labels)/n_proba

    for i, p_miss in enumerate(false_neg):
        if p_miss > p_allow_miss:
            return i-1
    return n_samples


def _speedup_merged(proba, labels, n_allow_miss=0, normalize=True):
    """ Merged version of _speedup(), compute average and mean. """
    speedup = []
    for sub_proba in proba:
        speedup.append(_speedup(sub_proba, labels, n_allow_miss, normalize))
    speed_avg = np.mean(speedup)
    if len(speedup) > 1:
        speed_err = stats.sem(speedup)
    else:
        speed_err = 0
    return [speed_avg, speed_err]


def _ROC(proba, labels):
    """ Compute the ROC of the prediction probabilities. """
    class_proba = _split_probabilities(proba, labels)
    num_c0 = class_proba[0].shape[0]
    num_c1 = class_proba[1].shape[0]

    y_score = np.concatenate((class_proba[0], class_proba[1]))
    np.nan_to_num(y_score, copy=False)
    y_true = np.concatenate((np.zeros(num_c0), np.ones(num_c1)))
    fpr, tpr, _ = roc_curve(y_true, y_score)

    roc_auc = auc(fpr, tpr)
    return roc_auc


def _ROC_merged(proba, labels):
    """ Merged version of _ROC(). """
    rocs = []
    for sub in proba:
        rocs.append(_ROC(sub, labels))
    roc_avg = np.mean(rocs)
    if len(rocs) > 1:
        roc_err = stats.sem(rocs)
    else:
        roc_err = 0
    return [roc_avg, roc_err]


def _inc_queried(proba, labels):
    """ Compute the number of queried labels that were one. """
    class_proba = _split_probabilities(proba, labels)
    num_pool1 = class_proba[1].shape[0]
    num_tot1 = sum(labels)
    return num_tot1-num_pool1


def _inc_queried_merged(proba, labels):
    """ Merged version of _inc_queried. """
    found = []
    for sub in proba:
        found.append(_inc_queried(sub, labels))
    found_avg = np.mean(found)
    if len(found) > 1:
        found_err = stats.sem(found)
    else:
        found_err = 0
    return [found_avg, found_err]


def _avg_proba(proba, labels):
    """ Average of the prediction probabilities. """
    n = len(proba)
    class_proba = _split_probabilities(proba, labels)
    results = []
    for i in range(2):
        new_mean = np.mean(class_proba[i])
        new_sem = stats.sem(class_proba[i])
        results.append([n, new_mean, new_sem])
    return results


def _avg_proba_merged(proba, labels):
    """ Merged version of prediction probabilities. """

    # Flatten list
    flat_proba = []
    for sub in proba:
        for item in sub:
            flat_proba.append(item)
    return _avg_proba(flat_proba, labels)


class Analysis(object):
    """ Analysis object to plot things from the logs. """

    def __init__(self, data_dirs):
        self._dirs = []
        self._results = {}
        self._rores = {}  # Reordered results.
        self._n_queries = {}
        self._labels = {}

        # Get the results for all the directories.
        for full_dir in data_dirs:
            _dir = os.path.normpath(full_dir)
            _dir = os.path.basename(_dir)
            self._dirs.append(_dir)
            self._results[_dir] = read_json_results(full_dir)
            self._rores[_dir] = reorder_results(self._results[_dir])
            self._n_queries[_dir] = get_num_queries(self._results[_dir])
            if 'labels' in list(self._results[_dir].values())[0]:
                self._labels[_dir] = list(self._results[_dir].values())[0]['labels']
            else:
                self._labels[_dir] = read_labels("pickle/schoot-lgmm-ptsd_words_20000.pkl")

    def _avg_time_found(self, _dir):
        results = self._rores[_dir]['labelled']
        time_results = {}
        res = {}
        num_query = len(results)
        n_labels = len(self._labels[_dir])
        n_queries = len(self._n_queries[_dir])

        for i, query in enumerate(results):
            n_queried = self._n_queries[_dir][i]
            n_files = len(results[query])
            for query_list in results[query]:
                for label_inc in query_list:
                    label = label_inc[0]
                    include = label_inc[1]
                    if not include:
                        continue
                    if label not in time_results:
                        time_results[label] = [0, 0, 0]
                    if i == 0:
                        time_results[label][2] += 1
                    else:
                        time_results[label][0] += n_queried
                        time_results[label][1] += 1
        penalty_not_found = 2*self._n_queries[_dir][n_queries-1] - self._n_queries[_dir][n_queries-2]
        for label in time_results:
            tres = time_results[label]
            n_not_found = n_files - tres[1] - tres[2]
#             print(n_not_found, penalty_not_found, n_files, tres[2])
            if n_files-tres[2]:
                res[label] = (n_not_found*penalty_not_found + tres[0])/(n_files-tres[2])
            else:
                res[label] = 0

        return res

    def print_avg_time_found(self):
        time_hist = []
        for _dir in self._dirs:
            res_dict = self._avg_time_found(_dir)
            res = list(res_dict.values())
            time_hist.append(res)
            for label in res_dict:
                if res_dict[label] > 1400:
                    print(f"{_dir}: label={label}, value={res_dict[label]}")
#             if time_hist is None:
#                 time_hist = np.array([res])
#             else:
#                 print((time_hist, np.array([res])))
#                 print(time_hist.shape)
#                 print(np.array([res]).shape)
#                 time_hist = np.concatenate((time_hist, np.array([res])))
#             print(time_hist)
#             time_hist = np.append(time_hist, np.array([res]), axis=1)
#         print(time_hist)
        plt.hist(time_hist, density=False)
        plt.show()

    def stat_test_merged(self, _dir, logname, stat_fn, **kwargs):
        """
        Do a statistical test on the results.

        Arguments
        ---------
        _dir: str
            Base directory key (path removed).
        logname: str
            Logname as given in the log file (e.g. "pool_proba").
        stat_fn: func
            Function to gather statistics, use merged_* version.
        kwargs: dict
            Extra keywords for the stat_fn function.

        Returns
        -------
        list:
            Results of the statistical test, format depends on stat_fn.
        """
        stat_results = []
        results = self._rores[_dir][logname]
#         print(self._results[_dir])
        labels = self._labels[_dir]
        for query in results:
            new_res = stat_fn(results[query], labels, **kwargs)
            stat_results.append(new_res)
        return stat_results

    def plot_ROC(self):
        """
        Plot the ROC for all directories and both the pool and
        the train set.
        """
        legend_name = []
        legend_plt = []
        pool_name = "pool_proba"
        label_name = "train_proba"
        for i, _dir in enumerate(self._dirs):
            pool_roc = self.stat_test_merged(
                _dir, pool_name, _ROC_merged)
            label_roc = self.stat_test_merged(
                _dir, label_name, _ROC_merged)
            cur_pool_roc = []
            cur_pool_err = []
            cur_label_roc = []
            cur_label_err = []
            xr = self._n_queries[_dir]
            for pool_data in pool_roc:
                cur_pool_roc.append(pool_data[0])
                cur_pool_err.append(pool_data[1])
            for label_data in label_roc:
                cur_label_roc.append(label_data[0])
                cur_label_err.append(label_data[1])

            col = "C"+str(i % 10)
            myplot = plt.errorbar(xr, cur_pool_roc, cur_pool_err, color=col)
            plt.errorbar(xr, cur_label_roc, cur_label_err, color=col, ls="--")
            legend_name.append(f"{_dir}")
            legend_plt.append(myplot)

        plt.legend(legend_plt, legend_name, loc="upper right")
        plt.title("Area Under Curve of ROC")
        plt.show()

    def plot_inc_found(self):
        """
        Plot the number of queries that turned out to be included
        in the final review.
        """
        legend_name = []
        legend_plt = []
        pool_name = "pool_proba"

        for i, _dir in enumerate(self._dirs):
            inc_found = self.stat_test_merged(_dir, pool_name,
                                              _inc_queried_merged)
            cur_inc_found = []
            cur_inc_found_err = []

            xr = self._n_queries[_dir]
            for inc_data in inc_found:
                cur_inc_found.append(inc_data[0])
                cur_inc_found_err.append(inc_data[1])

            dy = cur_inc_found[0]
            dx = self._n_queries[_dir][0]
            x_norm = (len(self._labels[_dir])-dx)/100
            y_norm = (np.sum(self._labels[_dir])-dy)/100

            col = "C"+str(i % 10)
            norm_xr = (np.array(xr)-dx)/x_norm
            norm_yr = (np.array(cur_inc_found)-dy)/y_norm
            norm_y_err = cur_inc_found_err/y_norm

            myplot = plt.errorbar(norm_xr, norm_yr, norm_y_err, color=col)
            legend_name.append(f"{_dir}")
            legend_plt.append(myplot)
#         print(x_norm, y_norm, dy)
#         start_inc = cur_pool_roc[0]
#         start_pool = (len(self._labels)-xr[0])
#         tot_inc = np.sum(self._labels)
#         y_random = [start_inc]
#         for i, x in enumerate(xr[1:]):
#             y = start_inc + (tot_inc-start_inc)*(x-xr[0])/start_pool
#             y_random.append(y)
#         my_plot, = plt.plot(np.array(xr)/x_norm, (np.array(y_random)-dy)/y_norm, color="black", ls="--")
#         legend_name.append("Random")
#         legend_plt.append(my_plot)
        plt.legend(legend_plt, legend_name, loc="upper left")
#         if normalize:
        symb = "%"
#         else:
#             symb = "#"
        plt.xlabel(f"{symb} Queries")
        plt.ylabel(f"< {symb} Inclusions queried >")
        plt.title("Average number of inclusions found")
        plt.grid()
        plt.show()

    def plot_proba(self):
        """
        Plot the average prediction probabilities of samples in the pool
        and training set.
        """
        pool_plt = []
        pool_leg_name = []
        label_plt = []
        label_leg_name = []
        legend_name = []
        legend_plt = []
        pool_name = "pool_proba"
        label_name = "train_proba"
        linestyles = ['-', '--', '-.', ':']

        for i, _dir in enumerate(self._dirs):
            pool_proba = self.stat_test_merged(
                _dir, pool_name, _avg_proba_merged)
            label_proba = self.stat_test_merged(
                _dir, label_name, _avg_proba_merged)
            col = "C"+str(i % 10)
            for true_cat in range(2):
                cur_pool_prob = []
                cur_pool_err = []
                cur_label_prob = []
                cur_label_err = []
                xr = self._n_queries[_dir]
                for pool_data in pool_proba:
                    cur_pool_prob.append(pool_data[true_cat][1])
                    cur_pool_err.append(pool_data[true_cat][2])
                for label_data in label_proba:
                    cur_label_prob.append(label_data[true_cat][1])
                    cur_label_err.append(label_data[true_cat][2])
                ls1 = linestyles[true_cat*2]
                ls2 = linestyles[true_cat*2+1]
                myplot = plt.errorbar(xr, cur_pool_prob, cur_pool_err,
                                      color=col, ls=ls1)
                myplot2 = plt.errorbar(xr, cur_label_prob, cur_label_err,
                                       color=col, ls=ls2)
                if i == 0:
                    pool_plt.append(myplot)
                    pool_leg_name.append(f"Pool: label = {true_cat}")
                    label_plt.append(myplot2)
                    label_leg_name.append(f"Train: label = {true_cat}")
                if true_cat == 1:
                    legend_plt.append(myplot)

            legend_name.append(f"{_dir}")
        legend_name += pool_leg_name
        legend_name += label_leg_name
        legend_plt += pool_plt
        legend_plt += label_plt
        plt.legend(legend_plt, legend_name, loc="upper right")
        plt.title("Probability of inclusion")
        plt.show()

    def plot_speedup(self, n_allow_miss=[0], normalize=False):
        """
        Plot the average number of papers that can be discarded safely.

        Arguments
        ---------
        n_allow_miss: list[int]
            A list of the number of allowed False Negatives.
        normalize: bool
            Normalize the output with the expected outcome (not correct).
        """
        legend_plt = []
        legend_name = []
        linestyles = ['-', '--', '-.', ':']

        for i, _dir in enumerate(self._dirs):
            for i_miss, n_miss in enumerate(n_allow_miss):
                speed_res = self.stat_test_merged(
                    _dir, "pool_proba", _speedup_merged,
                    n_allow_miss=n_miss, normalize=normalize)
                xr = self._n_queries[_dir]
                cur_avg = []
                cur_err = []
                for sp in speed_res:
                    cur_avg.append(sp[0])
                    cur_err.append(sp[1])
                col = "C"+str(i % 10)
                my_plot = plt.errorbar(xr, cur_avg, cur_err,
                                       color=col, capsize=4,
                                       ls=linestyles[i_miss % len(linestyles)])
                if n_miss == n_allow_miss[0]:
                    legend_plt.append(my_plot)
                    legend_name.append(f"{_dir}")

        plt.legend(legend_plt, legend_name, loc="upper left")
        plt.title("Articles that do not have to be read.")
        plt.show()

    def plot_limits(self, prob_allow_miss=[0.1]):
        legend_plt = []
        legend_name = []
        linestyles = ['-', '--', '-.', ':']

        for i, _dir in enumerate(self._dirs):
            for i_miss, p_miss in enumerate(prob_allow_miss):
                limits_res = self.stat_test_merged(
                    _dir, "pool_proba", _limits_merged,
                    p_allow_miss=p_miss)
                xr = self._n_queries[_dir]
                col = "C"+str(i % 10)
                my_plot, = plt.plot(xr, limits_res, color=col,
                                    ls=linestyles[i_miss % len(linestyles)])
                if i_miss == 0:
                    legend_plt.append(my_plot)
                    legend_name.append(f"{_dir}")

        plt.legend(legend_plt, legend_name, loc="upper left")
        plt.title("Articles that do not have to be read.")
        plt.show()

    