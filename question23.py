import numpy as np 
from utils.get_results import seed_all, run_multiple_simulations
from utils.plot import plot_rho_against_stat, Welch_test, plot_pvalues_heatmap

if __name__ == "__main__": 
    seed_all()

    num_servers_arr = np.array([1,2,4])
    rhos = np.linspace(0.01,1,100)
    mu = 1
    T = 2000
    num_runs_arr=np.linspace(2,200,100).astype(int)

    results_FIFO, results_SJF = run_multiple_simulations(num_runs_arr=num_runs_arr, rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, save_file=True)
    plot_rho_against_stat(results_FIFO, results_SJF, num_servers_arr, rhos, num_run_idx=-1, file_name='waiting_time_FIFO_MMN')

    X,Y = np.meshgrid(rhos, num_runs_arr)
    p_value_FIFO = Welch_test(results_FIFO, rhos, num_runs_arr, len(num_servers_arr))
    plot_pvalues_heatmap(X, Y, p_value_FIFO, num_servers_arr, file_name='pvalues_FIFO_MMN')

    p_value_SJF = Welch_test(results_SJF, rhos, num_runs_arr, len(num_servers_arr))
    plot_pvalues_heatmap(X, Y, p_value_SJF, num_servers_arr, file_name='pvalues_SJF_MMN')