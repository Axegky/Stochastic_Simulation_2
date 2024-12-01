import simpy
import numpy as np
from scipy.stats import expon

# Question 4: hyperexponential distribution for service time
def hyperexponential_rvs(mus: np.array, probs: np.array, n: int, random_state: int):
    """
        Generates n samples from a hyperexponential distribution with getting single sample x ~ exp(mus[i]) with probability probs[i].
    """

    choices = np.random.default_rng(random_state).choice(len(mus), size=n, p=probs)
    samples = np.zeros(n)

    for i, mu in enumerate(mus):
        mask = (choices == i)
        samples[mask] = expon.rvs(scale=1/mu, size=np.sum(mask), random_state=random_state)
    
    return samples

class MMN:
    def __init__(
        self, 
        rho: float, 
        mu: float, 
        num_servers_arr: np.array, 
        T: int, 
        random_state: int,
        SJF: bool = False,
        deterministic_service_time: float = None,
        hyperexp_service_time_params: dict = None
    ):
        """
            rho: the load of the system.
            mu: the capacity of each of the N servers.
            num_servers_arr: an array of different number of servers to run simulation for, 
                i.e. [1,2] denotes simulation is run with the same samples when there are 1 servers and 2 servers.
            T: the time horizon to run the simulation for.
            random_state: random_state used for seeding.
            SJF: If true, shortest job is first scheduled, otherwise, FIFO is used.
            deterministic_service_time: If not None, it is used as the deterministic service time.
            hyperexp_service_time_params: If not None, it is a dictionary containing:
                - 'mus': an array of rate parameters for the exponential distributions.
                - 'probs': an array of probabilities corresponding to each exponential distribution.
        """
    
        self.rho = rho
        self.mu = mu

        # Set the arrival rate self.lamb of the number of jobs into the system.
        if deterministic_service_time:
            self.lamb = rho * (num_servers_arr * deterministic_service_time)
        elif hyperexp_service_time_params:
            if 'mus' not in hyperexp_service_time_params or 'probs' not in hyperexp_service_time_params:
                raise ValueError("hyperexponential_service_time_params must contain 'mus' and 'probs' keys")
            
            self.mus = np.array(hyperexp_service_time_params['mus'])
            self.probs = np.array(hyperexp_service_time_params['probs'])
            self.lamb = rho * (num_servers_arr * (self.mus * self.probs).sum())
        else:
            self.lamb = rho * (num_servers_arr * mu)
    
        self.num_servers_arr = num_servers_arr
        self.T = T
        self.batch_sample_size = np.ceil(self.lamb*self.T*5).astype(int)
        self.num_diff_N = len(num_servers_arr)
        self.SJF = SJF
        self.deterministic_service_time = deterministic_service_time
        self.hyperexp_service_time_params = hyperexp_service_time_params

        # Reproducibility params
        self.random_state_lamb = random_state
        self.random_state_mu = random_state + 1
        self.random_state_jump = 10^4

        # SimPy environment and resources
        self.env_FIFO = simpy.Environment()
        self.env_SJF = simpy.Environment()
        self.servers_diff_N_FIFO = [simpy.Resource(self.env_FIFO, capacity=N) for N in num_servers_arr]
        self.servers_diff_N_SJF = [simpy.PriorityResource(self.env_SJF, capacity=N) for N in num_servers_arr]

        # Statistics
        self.waiting_times_FIFO = [[] for _ in range(self.num_diff_N)]
        self.system_times_FIFO = [[] for _ in range(self.num_diff_N)]
        self.waiting_times_SJF = [[] for _ in range(self.num_diff_N)]
        self.system_times_SJF = [[] for _ in range(self.num_diff_N)]

        # Get initial samples
        self.job_arrival_times_initial = [0]*self.num_diff_N
        self.service_durations_initial= [0]*self.num_diff_N
        for i in range(self.num_diff_N):
            self.job_arrival_times_initial[i], self.service_durations_initial[i] = self.__get_events(i, self.random_state_lamb, self.random_state_mu)

    def __get_events(self, Ns_idx, random_state_lamb, random_state_mu):
        lamb = self.lamb[Ns_idx]
        batch_sample_size = self.batch_sample_size[Ns_idx]
        job_arrival_times = expon.rvs(scale=1/lamb, size=batch_sample_size, random_state=random_state_lamb)
        service_durations = None
                
        # Since service duration is random, we check for whether hyperexponential dist is used for service duration.
        if self.hyperexp_service_time_params:
            service_durations = hyperexponential_rvs(self.mus, self.probs, batch_sample_size, random_state_mu)
        elif self.deterministic_service_time:
            pass
        else:
            service_durations = expon.rvs(scale=1/self.mu, size=batch_sample_size, random_state=random_state_mu)
        
        return job_arrival_times, service_durations

    def __job_arrival_FIFO(self, Ns_idx):
        """Generate job arrivals (FIFO)."""
        job_arrival_times = self.job_arrival_times_initial[Ns_idx]
        service_durations = self.service_durations_initial[Ns_idx]
        i = 0

        # Check if the service duration is random
        if not self.deterministic_service_time:
            while True:
                for job_arrival_time, service_duration in zip(job_arrival_times, service_durations):
                    yield self.env_FIFO.timeout(job_arrival_time)
                    arrival_time = self.env_FIFO.now
                    self.env_FIFO.process(self.__job_process_FIFO(service_duration, Ns_idx, arrival_time))
                
                i+=1
                random_state_lamb = self.random_state_lamb+i*self.random_state_jump
                random_state_mu = self.random_state_mu+i*self.random_state_jump
                job_arrival_times, service_durations = self.__get_events(Ns_idx, random_state_lamb, random_state_mu)
        else:
            while True:
                for job_arrival_time in job_arrival_times:
                    yield self.env_FIFO.timeout(job_arrival_time)
                    arrival_time = self.env_FIFO.now
                    self.env_FIFO.process(self.__job_process_FIFO(self.deterministic_service_time, Ns_idx, arrival_time))
                
                i+=1
                random_state_lamb = self.random_state_lamb+i*self.random_state_jump
                random_state_mu = self.random_state_mu+i*self.random_state_jump
                job_arrival_times, service_durations = self.__get_events(Ns_idx, random_state_lamb, random_state_mu)

    def __job_arrival_SJF(self, Ns_idx):
        """Generate job arrivals (SJF)."""
        job_arrival_times = self.job_arrival_times_initial[Ns_idx]
        service_durations = self.service_durations_initial[Ns_idx]
        i = 0

        # Check if the service duration is random
        if not self.deterministic_service_time:
            while True:
                for job_arrival_time, service_duration in zip(job_arrival_times, service_durations):
                    yield self.env_SJF.timeout(job_arrival_time)
                    arrival_time = self.env_SJF.now
                    self.env_SJF.process(self.__job_process_SJF(service_duration, Ns_idx, arrival_time))
                
                i+=1
                random_state_lamb = self.random_state_lamb+i*self.random_state_jump
                random_state_mu = self.random_state_mu+i*self.random_state_jump
                job_arrival_times, service_durations = self.__get_events(Ns_idx, random_state_lamb, random_state_mu)
        else:
            while True:
                for job_arrival_time in job_arrival_times:
                    yield self.env_SJF.timeout(job_arrival_time)
                    arrival_time = self.env_SJF.now
                    self.env_SJF.process(self.__job_process_SJF(self.deterministic_service_time, Ns_idx, arrival_time))
                
                i+=1
                random_state_lamb = self.random_state_lamb+i*self.random_state_jump
                random_state_mu = self.random_state_mu+i*self.random_state_jump
                job_arrival_times, service_durations = self.__get_events(Ns_idx, random_state_lamb, random_state_mu)
            
    def __job_process_FIFO(self, service_duration, Ns_idx, arrival_time):
        """Process a single job (FIFO)."""
    
        with self.servers_diff_N_FIFO[Ns_idx].request() as request:
            yield request

            self.waiting_times_FIFO[Ns_idx].append(self.env_FIFO.now - arrival_time)

            yield self.env_FIFO.timeout(service_duration)
            
            self.system_times_FIFO[Ns_idx].append(self.env_FIFO.now - arrival_time)
    
    def __job_process_SJF(self, service_duration, Ns_idx, arrival_time):
        """Process a single job (SJF)."""
        with self.servers_diff_N_SJF[Ns_idx].request(priority=service_duration) as request:
            yield request

            self.waiting_times_SJF[Ns_idx].append(self.env_SJF.now - arrival_time)

            yield self.env_SJF.timeout(service_duration)
            
            self.system_times_SJF[Ns_idx].append(self.env_SJF.now - arrival_time)
        
    def run_simulation(self):
        """Run the simulation."""
        # Run the simulation for different N at once consequently for both FIFO and SJF services.
        for Ns_idx in range(self.num_diff_N):
            self.env_FIFO.process(self.__job_arrival_FIFO(Ns_idx))
            self.env_FIFO.run(until=self.T*(Ns_idx+1))
        
        result_FIFO = {
            'avg_waiting_times': [np.mean(wt) for wt in self.waiting_times_FIFO],
            'avg_system_times': [np.mean(st) for st in self.system_times_FIFO],
            'waiting_times': self.waiting_times_FIFO,
            'system_times': self.system_times_FIFO
        }
            
        for Ns_idx in range(self.num_diff_N):
            self.env_SJF.process(self.__job_arrival_SJF(Ns_idx))
            self.env_SJF.run(until=self.T*(Ns_idx+1))

        result_SJF = {
            'avg_waiting_times': [np.mean(wt) for wt in self.waiting_times_SJF],
            'avg_system_times': [np.mean(st) for st in self.system_times_SJF],
            'waiting_times': self.waiting_times_SJF,
            'system_times': self.system_times_SJF
        }

        return result_FIFO, result_SJF
