from utils import load_json, compare_stats

stats_benchmark = load_json('results/stats_benchmark.json')
stats_target = load_json('results/stats_target.json')
stats_target_smaller_std = load_json('results/stats_target_smaller_std.json')
stats_target_larger_std = load_json('results/stats_target_larger_std.json')
stats_target_smaller_period = load_json('results/stats_target_smaller_std_smaller_period.json')
stats_target_larger_period = load_json('results/stats_target_smaller_std_larger_period.json')
stats_target_larger_dim = load_json('results/stats_target_larger_dim.json')

# compare_stats((stats_target, 'target'))
compare_stats(
    (stats_target, 'larger std'),
    (stats_target_smaller_std, 'target'),
    (stats_target_larger_std, 'huge std'),
    # (stats_target_smaller_period, 'smaller period'),
    # (stats_target_larger_period, 'larger period'),
    # (stats_target_larger_dim, 'larger dim'),
    # (stats_benchmark, 'benchmark')
)

# to do:
# 1. tune target model until convergence
#   a. (done) different settings of std, resample_period, reduced_dim
#   b. freeze pre-trained layers
#   c. time-varying std scheduler
#   d. no resample at all? why larger resample period converge better?
#   e. improve time performance
# 
# - how to manage experiments effectively?