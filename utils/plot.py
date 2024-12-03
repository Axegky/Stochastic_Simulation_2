import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import t
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def sns_lineplot(rhos, results, axs, num_servers_arr, Ns_Idx, service_type='FIFO', color = 'blue', num_run_idx=-1): 

    conf_waiting_time_lower = np.maximum(results['conf_waiting_time_lower'][:, num_run_idx, Ns_Idx], 0)

    # Plot average waiting time with confidence intervals
    sns.lineplot(x=rhos, y=results['avg_waiting_times'][:, num_run_idx, Ns_Idx], ax=axs, label=f'{num_servers_arr[Ns_Idx]}', marker='o', color=color)
    axs.fill_between(
        rhos,
        conf_waiting_time_lower,
        results['conf_waiting_time_upper'][:, num_run_idx, Ns_Idx],
        color=color, 
        alpha=0.2
    )

    axs.plot(rhos, conf_waiting_time_lower, color=color, linestyle='--', linewidth=1, alpha=0.5)
    axs.plot(rhos, results['conf_waiting_time_upper'][:, num_run_idx, Ns_Idx], color=color, linestyle='--', linewidth=1, alpha=0.5)

    axs.set_xlabel(r'$\rho$', fontsize=14)
    axs.set_ylabel(f'Mean Waiting Time', fontsize=13) 

    axs.legend(title='Number of Servers', fontsize=12)

    axs.set_title(f'{service_type}', fontsize=16)

def plot_rho_against_stat(
    results_FIFO,
    results_SJF, 
    num_servers_arr, 
    rhos,
    num_run_idx=-1,
    file_name=None
):
    num_diff_N = len(num_servers_arr)    
    _, axs = plt.subplots(1, 2, figsize=(11, 5), dpi=300, sharey=True, sharex=True)
    colors = ['orange', 'green', 'blue']

    for j in range(num_diff_N):
        sns_lineplot(rhos, results_FIFO, axs[0], num_servers_arr, j, color=colors[j], num_run_idx=num_run_idx)
        sns_lineplot(rhos, results_SJF, axs[1], num_servers_arr, j, 'SJF', color=colors[j], num_run_idx=num_run_idx)

    plt.tight_layout()

    if file_name:
        plt.savefig(f'figures/{file_name}.png')  
    else:
        plt.show()  

def Welch_test(results, rhos, num_runs_arr, num_diff_servers): 
    means = results['avg_waiting_times']
    stds = results['std_waiting_times']
    num_mcs = num_runs_arr[np.newaxis, :, np.newaxis]

    pvalues = np.zeros((num_diff_servers-1, len(num_runs_arr), len(rhos)))
    var_sum = stds[:, :, 0:1]**2 + stds[:, :, 1:num_diff_servers]**2
    t_stats = (means[:, :, 0:1] - means[:, :, 1:num_diff_servers]) / np.sqrt(var_sum / num_mcs)
    dof = (num_mcs - 1) * (var_sum**2) / (stds[:, :, 0:1]**2 + stds[:, :, 1:num_diff_servers]**2)
    pvalues = 1 - t.cdf(np.abs(t_stats), dof)

    return pvalues.transpose((2, 1, 0))

def plot_pvalues_heatmap(X, Y, pvalues, num_servers_array, file_name=None):
    fig, axs = plt.subplots(1, len(pvalues), figsize=(11, 5), dpi=300, sharex=True, sharey=True)    
    axs_flat = axs.flatten()
    titles = [fr'$\mathbb{{E}}[W(1)]$ and $\mathbb{{E}}[W({n})]$' for n in num_servers_array[1:]]
    
    cmap = plt.get_cmap('plasma_r') 
    cmap.set_under('black') 
    
    for ax, pvalue, title in zip(axs, pvalues, titles):
        heatmap = ax.contourf(X, Y, pvalue, cmap=cmap,  vmin=0.05, levels=20)  
        ax.set_title(title, fontsize=16)
        ax.grid()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    cbar = fig.colorbar(heatmap, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('p-value', labelpad=10)  
    
    axs_flat[0].set_xlabel(r'$\rho$', fontsize=14)
    axs_flat[1].set_xlabel(r'$\rho$', fontsize=14)
    axs_flat[0].set_ylabel('Number of Simulations', fontsize=13)
    
    plt.tight_layout() 
    plt.subplots_adjust(wspace=0.1, hspace=0.07, right=0.87)

    if file_name:
        plt.savefig(f'figures/{file_name}.png')  
    else:
        plt.show()  