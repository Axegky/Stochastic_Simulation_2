import os
from concurrent.futures import ThreadPoolExecutor

num_cpus = os.cpu_count()
print(f'Number of CPUs: {num_cpus}')

executor = ThreadPoolExecutor()
print(f"Number of worker threads: {executor._max_workers}")
