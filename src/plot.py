import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt


def conf_int(data, confidence=0.95):
    """inputs a list and a confidence, return the delta of the interval"""
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def explore_results(path, n_train):
    tests_r2 = []
    for i in range(n_train):
        with open(os.path.join(path, str(i), 'results.txt'), 'r') as f:
            test_r2 = f.readline().strip().split('\t')[1]
            tests_r2.append(float(test_r2))
    return tests_r2

results_dir = "results"
n_train = 10

mean = explore_results(os.path.join(results_dir, 'no_mirna', 'mean'), n_train)
LUAD = explore_results(os.path.join(results_dir, 'no_mirna', 'LUAD'), n_train)
LUSC = explore_results(os.path.join(results_dir, 'no_mirna', 'LUSC'), n_train)

mean_all = explore_results(os.path.join(results_dir, 'mirna', 'all', 'mean'), n_train)
LUAD_all = explore_results(os.path.join(results_dir, 'mirna', 'all', 'LUAD'), n_train)
LUSC_all = explore_results(os.path.join(results_dir, 'mirna', 'all', 'LUSC'), n_train)

mean_corr = explore_results(os.path.join(results_dir, 'mirna', 'corr', 'mean'), n_train)
LUAD_corr = explore_results(os.path.join(results_dir, 'mirna', 'corr', 'LUAD'), n_train)
LUSC_corr = explore_results(os.path.join(results_dir, 'mirna', 'corr', 'LUSC'), n_train)

mean_corr_abs = explore_results(os.path.join(results_dir, 'mirna', 'corr_abs', 'mean'), n_train)
LUAD_corr_abs = explore_results(os.path.join(results_dir, 'mirna', 'corr_abs', 'LUAD'), n_train)
LUSC_corr_abs = explore_results(os.path.join(results_dir, 'mirna', 'corr_abs', 'LUSC'), n_train)


df = pd.DataFrame(columns=['mean', 'LUAD', 'LUSC', 'mean_all', 'LUAD_all', 'LUSC_all', 'mean_corr', 'LUAD_corr', 'LUSC_corr', 'mean_corr_abs', 'LUAD_corr_abs', 'LUSC_corr_abs'])
df['mean'] = mean
df['LUAD'] = LUAD
df['LUSC'] = LUSC

df['mean_all'] = mean_all
df['LUAD_all'] = LUAD_all
df['LUSC_all'] = LUSC_all

df['mean_corr'] = mean_corr
df['LUAD_corr'] = LUAD_corr
df['LUSC_corr'] = LUSC_corr

df['mean_corr_abs'] = mean_corr_abs
df['LUAD_corr_abs'] = LUAD_corr_abs
df['LUSC_corr_abs'] = LUSC_corr_abs

print(df[['LUSC', 'LUSC_all', 'LUSC_corr', 'LUSC_corr_abs']])
df.to_csv('results/results.csv', sep='\t', index=False)


# print p-values
print('Xpresso vs all mirnas: ', scipy.stats.ttest_ind(mean, mean_all).pvalue)
print('Xpresso vs correlation: ', scipy.stats.ttest_ind(mean, mean_corr).pvalue)
print('Xpresso vs absolute value correlation: ', scipy.stats.ttest_ind(mean, mean_corr_abs).pvalue)

# print p-values
print('LUAD Xpresso vs all mirnas: ', scipy.stats.ttest_ind(LUAD, LUAD_all).pvalue)
print('LUAD Xpresso vs correlation: ', scipy.stats.ttest_ind(LUAD, LUAD_corr).pvalue)
print('LUAD Xpresso vs absolute value correlation: ', scipy.stats.ttest_ind(LUAD, LUAD_corr_abs).pvalue)

# print p-values
print('LUSC Xpresso vs all mirnas: ', scipy.stats.ttest_ind(LUSC, LUSC_all).pvalue)
print('LUSC Xpresso vs correlation: ', scipy.stats.ttest_ind(LUSC, LUSC_corr).pvalue)
print('LUSC Xpresso vs absolute value correlation: ', scipy.stats.ttest_ind(LUSC, LUSC_corr_abs).pvalue)



# GRAFICI Xpresso Vs all mirnas Vs corr Vs corr_abs
n_labels = 4
barWidth = 1/n_labels
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

Xpresso = [np.mean(mean), np.mean(LUAD), np.mean(LUSC)]
Xpresso_conf = [conf_int(mean), conf_int(LUAD), conf_int(LUSC)]
corr = [np.mean(mean_corr), np.mean(LUAD_corr), np.mean(LUSC_corr)]
corr_conf = [conf_int(mean_corr), conf_int(LUAD_corr), conf_int(LUSC_corr)]
corr_abs = [np.mean(mean_corr_abs), np.mean(LUAD_corr_abs), np.mean(LUSC_corr_abs)]
corr_abs_conf = [conf_int(mean_corr_abs), conf_int(LUAD_corr_abs), conf_int(LUSC_corr_abs)]
all_m = [np.mean(mean_all), np.mean(LUAD_all), np.mean(LUSC_all)]
all_m_conf = [conf_int(mean_all), conf_int(LUAD_all), conf_int(LUSC_all)]


br1 = np.arange(0, len(Xpresso)*(1+barWidth), 1+barWidth)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
plt.bar(br1, Xpresso, width=barWidth, label='Xpresso', edgecolor='black', yerr=Xpresso_conf)
plt.bar(br2, all_m, width=barWidth, label='all mirnas', edgecolor='black', yerr=all_m_conf)
plt.bar(br3, corr, width=barWidth, label='correlation', edgecolor='black', yerr=corr_conf)
plt.bar(br4, corr_abs, width=barWidth, label='absolute value correlation', edgecolor='black', yerr=corr_abs_conf)

# Adding Xticks
plt.xticks(np.arange(barWidth/2*(n_labels-1), len(Xpresso)*(1+barWidth), 1+barWidth), ['mean', 'LUAD', 'LUSC'])
plt.ylabel('$R^2$', rotation=0, labelpad=10)
plt.legend(framealpha=1)
plt.ylim([0.49, 0.6])
plt.grid(axis='y')
ax.set_axisbelow(True)
plt.show()
# save the figure in pdf format
plt.savefig('results/all_methods.pdf', dpi=400, bbox_inches='tight')
plt.close()