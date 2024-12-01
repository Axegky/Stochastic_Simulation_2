import numpy as np 

from get_results import seed_all, run_multiple_simulations, plot_rho_against_stat, Welch_test, plot_pvalues_heatmap

if __name__ == "__main__": 
    seed_all()
    
    num_servers_arr = np.array([1,2,4])
    # rhos=np.linspace(0.01,1,3)
    rhos=np.array([0.1,0.9])
    mu=None
    T=500
    num_runs=np.array([5,10])
    service_time=1

    results_MDN_FIFO, results_MDN_SJF = run_multiple_simulations(num_runs=num_runs, rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, deterministic_service_time=service_time, save_file=True)
    results_MDN_FIFO, results_MDN_SJF = run_multiple_simulations(num_runs=num_runs, rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, deterministic_service_time=service_time, load_file=True)

    plot_rho_against_stat(results_MDN_FIFO, results_MDN_SJF, num_servers_arr, rhos, num_run_idx=-1, file_name='waiting_time_FIFO_MDN')

    X,Y = np.meshgrid(num_runs, rhos)
    p_value_MDN_FIFO = Welch_test(results_MDN_FIFO, rhos, num_runs, len(num_servers_arr))
    plot_pvalues_heatmap(X, Y, p_value_MDN_FIFO, file_name='pvalues_FIFO_MDN')
    p_value_MDN_SJF = Welch_test(results_MDN_SJF, rhos, num_runs, len(num_servers_arr))
    plot_pvalues_heatmap(X, Y, p_value_MDN_SJF, file_name='pvalues_SJF_MDN')

