import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def chk_normality(data):
	groups = list(data.keys())
	for group in groups:
		stats.probplot(data[group], dist="norm", plot=plt)
		plt.title('Probability Plot - ' +  group)
		plt.show()
	return

def get_anova_table(data, alpha=0.05, tail_type='two_tailed', check_normality=False, check_homogeneity=False):
	""" This function returns ANOVA table (as a dictionary):
						| SS | df | MS | F | P_value | F_crit | P_decision | F_decision |
		[0]Among groups	|    |    |    |   |         |        |			   |            |
		[1]Within groups|    |    |    |   |         |        |			   |            |
		[2]Total        |    |    |    |   |         |        |			   |            |

		data: dictionary {'Group1':[values...], 'Group2':[values...], 'Group3':[values...], ...}
		alpha: level of significance (possible range= 0 to 1), default=0.05
		tail_type: tail hypothesis type (two_tailed, left_tailed, right_tailed), default=two_tailed
		return: dictionary {'SS':[among_groups, within_groups, total], 
							'df':[among_groups, within_groups, total],
							'MS':[among_groups, within_groups, total],
							'F':MS_among_groups/MS_within_groups,
							'P_value':P-value corresponding to F,
							'F_crit: F-critical corresponding to alpha, df_among_groups and df_within_groups' }
	"""
	if check_normality == True:
		chk_normality(data)

	anova_table = {}
	groups = list(data.keys())
	group_counts = []
	group_means = []
	group_stds = []
	all_data = []
	for group in groups:
		group_counts.append(len(data[group]))
		group_means.append(np.mean(data[group]))
		group_stds.append(np.std(data[group]))
		all_data.extend(data[group])
	if check_homogeneity == True:
		ratio = np.max(group_stds) / np.min(group_stds)
		if ratio < 2:
			print('max to min std ratio is ', ratio, ' and is < 2. So, homogeneity condition satisfied')
		else:
			print('max to min std ratio is ', ratio, ' and is > 2. So, homogeneity condition not satisfied')
	among_groups_counts = len(all_data)
	among_groups_mean = np.mean(all_data)
	among_groups_std = np.std(all_data)
	anova_table['df'] = [len(groups)-1, among_groups_counts-len(groups), among_groups_counts-1]
	SSTR = np.sum([group_counts[g]*(group_means[g]-among_groups_mean)**2 for g in range(len(groups))])
	SSE = np.sum([(group_counts[g]-1)*group_stds[g]**2 for g in range(len(groups))])
	SSTO = SSTR + SSE
	anova_table['SS'] = [SSTR, SSE, SSTO]
	anova_table['MS'] = [SSTR/anova_table['df'][0], SSE/anova_table['df'][1], SSTO/anova_table['df'][2]]
	F = anova_table['MS'][0] / anova_table['MS'][1]
	anova_table['F'] = F
	#calc p-value
	anova_table['P_value'] = 1 - stats.f.cdf(F, anova_table['df'][0], anova_table['df'][1])

	if tail_type == 'two_tailed':
		alpha /= 2

	anova_table['F_crit'] = stats.f.ppf(1-alpha, anova_table['df'][0], anova_table['df'][1])

	# The p-value approach
	print('Approach 1: The p-value approach to hypothesis testing in the decision rule')
	conclusion = 'Failed to reject the null hypothesis.'
	if anova_table['P_value'] <= alpha:
		conclusion = 'Null Hypothesis is rejected.'
	print('F-score is:', anova_table['F'], ' and p value is:', anova_table['P_value'])    
	print(conclusion)
	anova_table['P_decision'] = conclusion

	# The critical value approach
	print('\n--------------------------------------------------------------------------------------')
	print('Approach 2: The critical value approach to hypothesis testing in the decision rule')
	conclusion = 'Failed to reject the null hypothesis.'
	if anova_table['F'] > anova_table['F_crit']:
		conclusion = 'Null Hypothesis is rejected.'
	print('F-score is:', anova_table['F'], ' and critical value is:', anova_table['F_crit'])
	print(conclusion)  	
	anova_table['F_decision'] = conclusion
	
	return anova_table