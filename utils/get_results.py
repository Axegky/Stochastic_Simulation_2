from MMN import MMN
import numpy as np
from scipy.stats import t
from concurrent.futures import ProcessPoolExecutor
import random
import pickle 
from functools import partial

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

def run_sim(rho_num_mc_idx, mu, num_servers_arr, T, random_state_offset, deterministic_service_time, hyperexp_service_time_params):
    return MMN(
        rho=rho_num_mc_idx[0],
        mu=mu,
        num_servers_arr=num_servers_arr,
        T=T,
        random_state=random_state_offset + rho_num_mc_idx[1],
        deterministic_service_time=deterministic_service_time,
        hyperexp_service_time_params=hyperexp_service_time_params
    ).run_simulation()

def process_results(results, sim_res_mc, num_mc_idx, rho_idx, num_diff_servers):
    """Helper function to process results for FIFO or SJF."""
    for sim_res in sim_res_mc:
        for key in sim_res:
            if key.startswith('avg_'):
                results[key][rho_idx, num_mc_idx:] += sim_res[key]
            else:
                for N_Idx in range(num_diff_servers):
                    wt = results[key][rho_idx][N_Idx]
                    wt.extend(sim_res[key][N_Idx])
                    results[f'std_{key}'][rho_idx, num_mc_idx, N_Idx] = np.std(wt, ddof=1)

def postprocess_results(results, num_runs_arr):
    for key in results:
        if key.startswith('avg_'):
            results[key] /= num_runs_arr[np.newaxis, :, np.newaxis]

    t_ppfs = t.ppf(0.975, num_runs_arr - 1)[np.newaxis, :, np.newaxis]
    sqrt_num_runs = np.sqrt(num_runs_arr)[np.newaxis, :, np.newaxis]

    conf_margin_waiting_time = results['std_waiting_times']*t_ppfs/sqrt_num_runs

    # Store confidence intervals
    results['conf_waiting_time_upper'] = results['avg_waiting_times'] + conf_margin_waiting_time
    results['conf_waiting_time_lower'] = results['avg_waiting_times'] - conf_margin_waiting_time

def run_multiple_simulations(
        num_runs_arr: np.array, 
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
    num_diff_run = len(num_runs_arr)
    max_num_run = np.max(num_runs_arr).astype(int)
    smallest_num_run = np.min(num_runs_arr)
    
    file_prefix = f'rho_{smallest_rho}_to_{largest_rho}_with_num_{num_rhos}_mu_{mu}_T_{T}_num_runs_{smallest_num_run}_to_{max_num_run}_with_num_{num_diff_run}_'

    results_FIFO = {
        'avg_waiting_times': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float32),
        'std_waiting_times': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float32),
        'waiting_times': [[[] for _ in range(num_diff_N)] for _ in range(num_rhos)]
    }

    results_SJF = {
        'avg_waiting_times': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float32),
        'std_waiting_times': np.zeros((num_rhos, num_diff_run, num_diff_N), dtype=np.float32),
        'waiting_times': [[[] for _ in range(num_diff_N)] for _ in range(num_rhos)]
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

    run_sim_partial = partial(run_sim, mu=mu, num_servers_arr=num_servers_arr, T=T, random_state_offset=random_state_offset, deterministic_service_time=deterministic_service_time, hyperexp_service_time_params=hyperexp_service_time_params)

    print(f'Starting Simulation with {max_num_run} runs on {num_rhos} different rhos...')

    results_diff_rhos_runs = None
    rho_mc_idxs = [(rho, mc_idx) \
            for rho in rhos 
            for mc_idx in range(max_num_run)]

    # We use random_state to seed parallel programming
    with ProcessPoolExecutor() as ex:
        results_diff_rhos_runs = list(ex.map(run_sim_partial, rho_mc_idxs))

    print('Finished Simulation!')
    print('Processing Results...')

    process_results_partial = partial(process_results, num_diff_servers=num_diff_N)
    postprocess_results_partial = partial(postprocess_results, num_runs_arr=num_runs_arr)

    num_runs_arr_with_zero = np.concatenate(([0], num_runs_arr))

    for rho_idx in range(num_rhos):
        results_rho_diff_runs = results_diff_rhos_runs[rho_idx*max_num_run:(rho_idx+1)*max_num_run]
        results_rho_diff_runs_FIFO = [res[0] for res in results_rho_diff_runs]
        results_rho_diff_runs_SJF = [res[1] for res in results_rho_diff_runs]
        for num_mc_idx, num_mc in enumerate(num_runs_arr):
            start_idx = num_runs_arr_with_zero[num_mc_idx]
            sim_res_MC_FIFO = results_rho_diff_runs_FIFO[start_idx:num_mc]
            sim_res_MC_SJF = results_rho_diff_runs_SJF[start_idx:num_mc]

            # Permutable objects results_FIFO and results_SJF are changed in the function
            process_results_partial(results_FIFO, sim_res_MC_FIFO, num_mc_idx, rho_idx)
            process_results_partial(results_SJF, sim_res_MC_SJF, num_mc_idx, rho_idx)

    del results_FIFO['waiting_times']
    del results_SJF['waiting_times']

    print('Finished Processing Results!')
    print('Postprocessing Results...')

    postprocess_results_partial(results_FIFO)
    postprocess_results_partial(results_SJF)
    
    print('Finished Postprocessing Results!')

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