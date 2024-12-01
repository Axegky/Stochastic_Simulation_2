import numpy as np 
from get_results import seed_all, run_multiple_simulations, plot_rho_against_stat, Welch_test, plot_pvalues_heatmap

if __name__ == "__main__": 
    seed_all()

    num_servers_arr = np.array([1,2,4])
    rhos=np.array([0.1,0.2])
    mu = 1
    T=100
    num_runs=np.array([1,2])

    results_FIFO, results_SJF = run_multiple_simulations(num_runs=num_runs, rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, save_file=True)
    results_FIFO, results_SJF = run_multiple_simulations(num_runs=num_runs, rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, load_file=True)

    plot_rho_against_stat(results_FIFO, results_SJF, num_servers_arr, rhos, num_run_idx=-1, file_name='waiting_time_FIFO_MMN')

    X,Y = np.meshgrid(num_runs, rhos)
    p_value_FIFO = Welch_test(results_FIFO, rhos, num_runs, len(num_servers_arr))
    plot_pvalues_heatmap(X, Y, p_value_FIFO, file_name='pvalues_FIFO_MMN')

    p_value_SJF = Welch_test(results_SJF, rhos, num_runs, len(num_servers_arr))
    plot_pvalues_heatmap(X, Y, p_value_SJF, file_name='pvalues_SJF_MMN')