import numpy as np 
from get_results import seed_all, run_multiple_simulations, plot_rho_against_stat, Welch_test, plot_pvalues_heatmap

if __name__ == "__main__": 
    seed_all()

    num_servers_arr = np.array([1,2,4])
    rhos = np.linspace(0.02,1,2)
    mu = 1
    T = 1000
    num_runs = np.zeros((5,10))
    for i in range(5): 
        num_runs[i,:]=np.linspace(20*i+2,20*(i+1),10).astype(int)

    results_FIFO, results_SJF = run_multiple_simulations(num_runs=np.array([2,3]), rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, save_file=True)
    results_FIFO_1, results_SJF_1 = run_multiple_simulations(num_runs=num_runs[1], rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, save_file=True)
    results_FIFO_2, results_SJF_2 = run_multiple_simulations(num_runs=num_runs[2], rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, save_file=True)
    results_FIFO_3, results_SJF_3 = run_multiple_simulations(num_runs=num_runs[3], rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, save_file=True)
    results_FIFO_4, results_SJF_4 = run_multiple_simulations(num_runs=num_runs[4], rhos=rhos, mu=mu, num_servers_arr=num_servers_arr, T=T, save_file=True)

    dicts_to_combine = [results_FIFO_1, results_FIFO_2, results_FIFO_3, results_FIFO_4, results_SJF_1, results_SJF_2, results_SJF_3, results_SJF_4]
    for d in dicts_to_combine[:4]:
        for key, array in d.items():
            if key in results_FIFO:
                results_FIFO[key] = np.concatenate(results_FIFO[key], d[key], axis=1)


    # plot_rho_against_stat(results_FIFO, results_SJF, num_servers_arr, rhos, num_run_idx=-1, file_name='waiting_time_FIFO_MMN')

    # X,Y = np.meshgrid(num_runs, rhos)
    # p_value_FIFO = Welch_test(results_FIFO, rhos, num_runs, len(num_servers_arr))
    # plot_pvalues_heatmap(X, Y, p_value_FIFO, num_servers_arr, file_name='pvalues_FIFO_MMN')

    # p_value_SJF = Welch_test(results_SJF, rhos, num_runs, len(num_servers_arr))
    # plot_pvalues_heatmap(X, Y, p_value_SJF, num_servers_arr, file_name='pvalues_SJF_MMN')