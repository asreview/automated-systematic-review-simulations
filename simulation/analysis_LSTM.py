#!/usr/bin/env python

import sys
from pargrid import Analysis


args = sys.argv[1:]
if len(args) > 0:
    json_dirs = args
else:
    json_dirs = ["output"]


my_analysis = Analysis(json_dirs)
# my_analysis.plot_proba()
# my_analysis.plot_speedup([0, 1, 2, 3], normalize=False)
my_analysis.plot_inc_found("inc_res.json")
# my_analysis.plot_ROC()
# my_analysis.plot_limits([0.1, 0.5, 1, 2])
# my_analysis.print_avg_time_found()
