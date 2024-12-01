from MMN import MMN
import numpy as np
from scipy.stats import expon, t
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import random
import seaborn as sns
import pickle 
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def convert_to_float32(results):
    results_32 = {}
    for key in results:
        if key.startswith('conf_') or key.startswith('avg_'):
            results_32[key] = results[key].astype(float)
        else:
            results_32[key] = results[key]
    
    return results_32

def run_multiple_simulations(
        num_runs: np.array, 
        rhos: np.array, 
        mu: float, 
        num_servers_arr: np.array, 
        T: int, 
        random_state_offset=0,
        save_file=None, 
        load_file=None,
        deterministic_service_time=None,
        hyperexp_service_time_params=None
    ):
    """Run multiple simulations and average the results."""

    seed_all()

    num_diff_N = len(num_servers_arr)
    smallest_rho = np.min(rhos)
    largest_rho = np.max(rhos)
    num_rhos = len(rhos)
    num_diff_run = len(num_runs)
    max_num_run = np.max(num_runs)
    smallest_num_run = np.min(num_runs)
    
    file_prefix = f'rho_{smallest_rho}_to_{largest_rho}_with_num_{num_rhos}_mu_{mu}_T_{T}_num_runs_{smallest_num_run}_to_{max_num_run}_with_num_{num_diff_run}_'

    results_FIFO = {
        'avg_waiting_times': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'avg_system_times': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'conf_waiting_time_upper': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'conf_waiting_time_lower': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'conf_system_time_upper': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'conf_system_time_lower': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'waiting_times': [[[] for _ in range(num_diff_N)] for _ in range(num_rhos)],
        'system_times': [[[] for _ in range(num_diff_N)] for _ in range(num_rhos)]
    }

    results_SJF = {
        'avg_waiting_times': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'avg_system_times': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'conf_waiting_time_upper': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'conf_waiting_time_lower': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'conf_system_time_upper': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'conf_system_time_lower': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float16),
        'waiting_times': [[[] for _ in range(num_diff_N)] for _ in range(num_rhos)],
        'system_times': [[[] for _ in range(num_diff_N)] for _ in range(num_rhos)]
    }

    if load_file:
        if deterministic_service_time:
            with open(f'data/{file_prefix}SJF_D.pkl', 'rb') as f:
                results_SJF = pickle.load(f)
            with open(f'data/{file_prefix}FIFO_D.pkl', 'rb') as f:
                results_FIFO = pickle.load(f)
        elif hyperexp_service_time_params: 
            with open(f'data/{file_prefix}SJF_H.pkl', 'rb') as f:
                results_SJF = pickle.load(f)
            with open(f'data/{file_prefix}FIFO_H.pkl', 'rb') as f:
                results_FIFO = pickle.load(f)
        else: 
            with open(f'data/{file_prefix}SJF_M.pkl', 'rb') as f:
                results_SJF = pickle.load(f)
            with open(f'data/{file_prefix}FIFO_M.pkl', 'rb') as f:
                results_FIFO = pickle.load(f)
        
        results_FIFO = convert_to_float32(results_FIFO)
        results_SJF = convert_to_float32(results_SJF)

        return results_FIFO, results_SJF

    rho_MC_idxs = [(rho_idx, i) for i in range(max_num_run) for rho_idx in range(num_rhos)]

    MMNs = [
        MMN(
            rho=rhos[rho_MC_idx[0]],
            mu=mu,
            num_servers_arr=np.array(num_servers_arr),
            T=T,
            random_state=random_state_offset + rho_MC_idx[1],
            deterministic_service_time=deterministic_service_time,
            hyperexp_service_time_params=hyperexp_service_time_params
        )
        for rho_MC_idx in rho_MC_idxs
    ]

    def run_simulation(idxs):
        rho_idx, mc_idx = idxs
        return MMNs[rho_idx * max_num_run + mc_idx].run_simulation()

    results_diff_rhos_runs = None

    # We use random_state to seed parallel programming
    with ThreadPoolExecutor() as ex:
        results_diff_rhos_runs = list(ex.map(run_simulation, rho_MC_idxs))
    
    def process_results(results, sim_res, rho_idx, num_mc, num_servers_arr, num_mc_idx):
        """Helper function to process results for FIFO or SJF."""
        for key in sim_res:
            if key.startswith('avg_'):
                results[key][rho_idx, num_mc_idx] += sim_res[key]
            elif key.startswith('conf_'):
                pass
            else:
                for N_Idx in range(len(num_servers_arr)):
                    results[key][rho_idx][N_Idx].extend(sim_res[key][N_Idx])

        for key in results:
            if key.startswith('avg_'):
                results[key][rho_idx] /= num_mc
        
        conf_margin_waiting_time = [
            np.std(waiting_times, ddof=1) * t.ppf(0.975, num_mc - 1) / np.sqrt(num_mc)
            for waiting_times in results['waiting_times'][rho_idx]
        ]
        conf_margin_system_time = [
            np.std(system_times, ddof=1) * t.ppf(0.975, num_mc - 1) / np.sqrt(num_mc)
            for system_times in results['system_times'][rho_idx]
        ]

        # Store confidence intervals
        results['conf_waiting_time_upper'][rho_idx, num_mc_idx] = [
            results['avg_waiting_times'][rho_idx, num_mc_idx][i] + conf_margin
            for i, conf_margin in enumerate(conf_margin_waiting_time)
        ]
        results['conf_waiting_time_lower'][rho_idx, num_mc_idx] = [
            results['avg_waiting_times'][rho_idx, num_mc_idx][i] - conf_margin
            for i, conf_margin in enumerate(conf_margin_waiting_time)
        ]

        results['conf_system_time_upper'][rho_idx, num_mc_idx] = [
            results['avg_system_times'][rho_idx, num_mc_idx][i] + conf_margin
            for i, conf_margin in enumerate(conf_margin_system_time)
        ]
        results['conf_system_time_lower'][rho_idx, num_mc_idx] = [
            results['avg_system_times'][rho_idx, num_mc_idx][i] - conf_margin
            for i, conf_margin in enumerate(conf_margin_system_time)
        ]

        return results

    for rho_idx in range(num_rhos):
        for num_mc_idx, num_mc in enumerate(num_runs):
            results_diff_runs = results_diff_rhos_runs[rho_idx*max_num_run:rho_idx*max_num_run+num_mc+1]
            for sim_res_FIFO, sim_res_SJF in results_diff_runs:
                results_FIFO = process_results(results_FIFO, sim_res_FIFO, rho_idx, num_mc, num_servers_arr, num_mc_idx)
                results_SJF = process_results(results_SJF, sim_res_SJF, rho_idx, num_mc, num_servers_arr,num_mc_idx)

    
    if save_file:
        if deterministic_service_time:
            with open(f'data/{file_prefix}FIFO_D.pkl', 'wb') as f:
                pickle.dump(results_FIFO, f)
            with open(f'data/{file_prefix}SJF_D.pkl', 'wb') as f:
                pickle.dump(results_SJF, f)
        elif hyperexp_service_time_params: 
            with open(f'data/{file_prefix}FIFO_H.pkl', 'wb') as f:
                pickle.dump(results_FIFO, f)
            with open(f'data/{file_prefix}SJF_H.pkl', 'wb') as f:
                pickle.dump(results_SJF, f)
        else: 
            with open(f'data/{file_prefix}FIFO_M.pkl', 'wb') as f:
                pickle.dump(results_FIFO, f)
            with open(f'data/{file_prefix}SJF_M.pkl', 'wb') as f:
                pickle.dump(results_SJF, f)
            
    results_FIFO = convert_to_float32(results_FIFO)
    results_SJF = convert_to_float32(results_SJF)

    return results_FIFO, results_SJF

def sns_lineplot(rhos, results, axs, num_servers_arr, Ns_Idx, service_type='FIFO', color = 'blue', num_run_idx=-1): 

    conf_waiting_time_lower = np.maximum(results['conf_waiting_time_lower'][:, num_run_idx, Ns_Idx], 0)
    conf_system_time_lower = np.maximum(results['conf_system_time_lower'][:, num_run_idx, Ns_Idx], 0)

    # Plot average waiting time with confidence intervals
    sns.lineplot(x=rhos, y=results['avg_waiting_times'][:, num_run_idx, Ns_Idx], ax=axs[0], label=f'{num_servers_arr[Ns_Idx]}', marker='o', color=color)
    axs[0].fill_between(
        rhos,
        conf_waiting_time_lower,
        results['conf_waiting_time_upper'][:, num_run_idx, Ns_Idx],
        color=color, 
        alpha=0.2
    )

    axs[0].plot(rhos, conf_waiting_time_lower, color=color, linestyle='--', linewidth=1, alpha=0.5)
    axs[0].plot(rhos, results['conf_waiting_time_upper'][:, num_run_idx, Ns_Idx], color=color, linestyle='--', linewidth=1, alpha=0.5)

    # Plot average system time with confidence intervals
    sns.lineplot(x=rhos, y=results['avg_system_times'][:, num_run_idx, Ns_Idx], ax=axs[1], label=f'{num_servers_arr[Ns_Idx]}', marker='o', color=color)
    axs[1].fill_between(
        rhos, 
        conf_system_time_lower, 
        results['conf_system_time_upper'][:, num_run_idx, Ns_Idx], 
        color=color,
        alpha=0.2
    )

    axs[1].plot(rhos, conf_system_time_lower, color=color, linestyle='--', linewidth=1, alpha=0.5)
    axs[1].plot(rhos, results['conf_system_time_upper'][:, num_run_idx, Ns_Idx], color=color, linestyle='--', linewidth=1, alpha=0.5)

    axs[0].set_xlabel(r'$\rho$', fontsize=14)
    axs[0].set_ylabel(f'Mean Waiting Time', fontsize=14) 
    axs[1].set_xlabel(r'$\rho$', fontsize=14)
    axs[1].set_ylabel(f'Mean System Time', fontsize=14)

    # Set legends for each subplot
    axs[0].legend(title='Number of Servers', fontsize=12)
    axs[1].legend(title='Number of Servers', fontsize=12)

    axs[0].set_title(f'{service_type}', fontsize=16)

def plot_rho_against_stat(
    results_FIFO,
    results_SJF, 
    num_servers_arr, 
    rhos,
    num_run_idx=-1,
    file_name=None
):
    num_diff_N = len(num_servers_arr)    
    _, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=300, sharey=True, sharex=True)
    colors = ['orange', 'green', 'blue']

    for j in range(num_diff_N):
        sns_lineplot(rhos, results_FIFO, axs[:, 0], num_servers_arr, j, color=colors[j], num_run_idx=num_run_idx)
        sns_lineplot(rhos, results_SJF, axs[:, 1], num_servers_arr, j, 'SJF', color=colors[j], num_run_idx=num_run_idx)

    plt.tight_layout()

    if file_name:
        plt.savefig(f'figures/{file_name}.png')  
    else:
        plt.show()  

def Welch_test(results, rhos, num_runs, num_diff_servers): 
    pvalues = np.zeros((num_diff_servers-1, len(num_runs), len(rhos)))
    for num_mc_idx, num_mc in enumerate(num_runs): 
        for rho_idx, _ in enumerate(rhos):
            means = results['avg_waiting_times'][rho_idx, num_mc_idx]
            stds = (results['conf_waiting_time_upper'][rho_idx, num_mc_idx, 0] - means) / t.ppf(0.975, num_mc - 1) * np.sqrt(num_mc)
            for i in range(num_diff_servers-1):
                var_sum = stds[0]**2 + stds[i+1]**2
                t_stat = (means[0]-means[i+1]) / np.sqrt(var_sum / num_mc)
                dof = (num_mc-1)*(var_sum**2) / (stds[0]**2 + stds[i+1]**2)
                pvalues[i, num_mc_idx, rho_idx] = 1 - t.cdf(np.abs(t_stat), dof)

    return pvalues

def plot_pvalues_heatmap(X, Y, pvalues, file_name=None):
    num_plots = len(pvalues)
    fig, axs = plt.subplots(1, num_plots, figsize=(16, 5), dpi=300, sharex=True, sharey=True)    
    axs_flat = axs.flatten()
    titles = ['n = 1, 2', 'n = 1, 4']
    cmap = plt.get_cmap('plasma_r') 
    cmap.set_under('black') 

    cbarticks=np.arange(0.0,1.0,0.1)

    for ax, pvalue, title in zip(axs, pvalues, titles):
        ax.set_facecolor('darkgrey')
        heatmap = ax.contourf(X, Y, pvalue, cbarticks, cmap=cmap, vmin=0.05, vmax=1, levels=20)
        ax.set_xlabel('Number of runs', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid()
    
    cbar = fig.colorbar(heatmap, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('p-value')
    
    axs_flat[0].set_ylabel(r'$\rho$', fontsize=14)
    plt.tight_layout() 

    if file_name:
        plt.savefig(f'figures/{file_name}.png')  
    else:
        plt.show()  