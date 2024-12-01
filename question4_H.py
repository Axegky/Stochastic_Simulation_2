import numpy as np 

from get_results import seed_all, run_multiple_simulations, plot_rho_against_stat, Welch_test, plot_pvalues_heatmap

if __name__ == "__main__": 
    seed_all()
    
    num_servers_arr = np.array([1,2,4])
    rhos = np.linspace(0.02,1,50)
    mu = None
    T = 1000
    num_runs=np.linspace(2,100,50)
   
    hyperexp_service_time_params={
        'mus': np.array([0.25, 1.5]),
        'probs': np.array([0.4, 0.6])
    }

    results_hyperexp_FIFO, results_hyperexp_SJF = run_multiple_simulations(num_runs=num_runs, rhos=rhos, mu=mu, num_servers_arr=np.array(num_servers_arr), T=T, hyperexp_service_time_params=hyperexp_service_time_params, save_file=True)
    results_hyperexp_FIFO, results_hyperexp_SJF = run_multiple_simulations(num_runs=num_runs, rhos=rhos, mu=mu, num_servers_arr=np.array(num_servers_arr), T=T, hyperexp_service_time_params=hyperexp_service_time_params, load_file=True)

    plot_rho_against_stat(results_hyperexp_FIFO, results_hyperexp_SJF, num_servers_arr, rhos, num_run_idx=-1, file_name='waiting_time_FIFO_MHN')

    X,Y = np.meshgrid(num_runs, rhos)
    p_value_hyperexp_FIFO = Welch_test(results_hyperexp_FIFO, rhos, num_runs, len(num_servers_arr))
    plot_pvalues_heatmap(X, Y, p_value_hyperexp_FIFO, num_servers_arr, file_name='pvalues_FIFO_MHN')