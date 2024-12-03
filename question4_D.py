import numpy as np 
from utils.get_results import seed_all, run_multiple_simulations
from utils.plot import plot_rho_against_stat, Welch_test, plot_pvalues_heatmap

if __name__ == "__main__": 
    seed_all()
    
    mu = None

    num_servers_arr = np.array([1,2,4])
    rhos = np.linspace(0.01,1,100)
    T = 2000
    max_num_run = 200
    num_runs_arr=np.linspace(2,max_num_run,100).astype(int)
    service_time = 1

    results_MDN_FIFO, results_MDN_SJF = run_multiple_simulations(num_runs_arr=num_runs_arr, rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, deterministic_service_time=service_time, save_file=True)

    plot_rho_against_stat(results_MDN_FIFO, results_MDN_SJF, num_servers_arr, rhos, num_run_idx=-1, file_name='waiting_time_FIFO_MDN')

    X,Y = np.meshgrid(rhos, num_runs_arr)
    p_value_MDN_FIFO = Welch_test(results_MDN_FIFO, rhos, num_runs_arr, len(num_servers_arr))
    plot_pvalues_heatmap(X, Y, p_value_MDN_FIFO, num_servers_arr, file_name='pvalues_FIFO_MDN')
    p_value_MDN_SJF = Welch_test(results_MDN_SJF, rhos, num_runs_arr, len(num_servers_arr))
    plot_pvalues_heatmap(X, Y, p_value_MDN_SJF, num_servers_arr, file_name='pvalues_SJF_MDN')

