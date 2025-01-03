import numpy as np 
from utils.get_results import seed_all, run_multiple_simulations
from utils.plot import plot_rho_against_stat, Welch_test, plot_pvalues_heatmap

if __name__ == "__main__": 
    seed_all()
    
    # Set parameters for M/H/n queue
    mu = None

    num_servers_arr = np.array([1,2,4])
    rhos = np.linspace(0.01,1,100)
    T = 2000
    max_num_run = 200
    num_runs_arr=np.linspace(2,max_num_run,100).astype(int)
    params={
        'mus': np.array([2/3, 1.5]),
        'probs': np.array([0.4, 0.6])
    }

    # Run simulation for M/H/n queue with FIFO, SJD
    results_hyperexp_FIFO, results_hyperexp_SJF = run_multiple_simulations(num_runs_arr=num_runs_arr, rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, hyperexp_service_time_params=params, save_file=True)
    # Plot the line graph of M/H/n queue w.r.t. rho
    plot_rho_against_stat(results_hyperexp_FIFO, results_hyperexp_SJF, num_servers_arr, rhos, num_run_idx=-1, file_name='waiting_time_FIFO_MHN')

    # Welch-test for E[W(1)]=E[W(2)], E[W(1)]=E[W(4)]
    X,Y = np.meshgrid(rhos, num_runs_arr)
    # test for M/H/n queue wirh FIFO type
    p_value_hyperexp_FIFO = Welch_test(results_hyperexp_FIFO, rhos, num_runs_arr, len(num_servers_arr))
    # Plot the p-value of Welch-test for M/H/n with FIFO type
    plot_pvalues_heatmap(X, Y, p_value_hyperexp_FIFO, num_servers_arr, file_name='pvalues_FIFO_MHN')

    # test for M/H/n queue wirh SJF type
    p_value_hyperexp_SJF = Welch_test(results_hyperexp_SJF, rhos, num_runs_arr, len(num_servers_arr))
    # Plot the p-value of Welch-test for M/H/n with SJF type
    plot_pvalues_heatmap(X, Y, p_value_hyperexp_SJF, num_servers_arr, file_name='pvalues_SJF_MHN')