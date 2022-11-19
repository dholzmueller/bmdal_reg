import time
import dill
import torch.multiprocessing as mp
import os
import sys
import traceback

from .train import *


def get_cuda_devices():
    return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]


def get_devices(use_gpu=True):
    return get_cuda_devices() if (use_gpu and torch.cuda.is_available()) else [torch.device('cpu')]


class GPUInfo:
    """
    Internal class.
    Provides some static methods that can be used to get information about (NVidia) GPUs.
    """
    _initialized = False

    @staticmethod
    def init():
        if not GPUInfo._initialized:
            import pynvml
            pynvml.nvmlInit()
            GPUInfo._initialized = True

    @staticmethod
    def get_process_ram_usage_gb(gpu_num: int, pid: int) -> float:
        # see https://github.com/gpuopenanalytics/pynvml/issues/21
        GPUInfo.init()
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_num)
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == pid:
                return proc.usedGpuMemory / (1024. ** 3)
        else:
            return 0.0

    @staticmethod
    def get_total_ram_gb(gpu_num: int) -> float:
        GPUInfo.init()
        import pynvml
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_num)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        total = info.total
        return total / (1024. ** 3)


class DeviceInfo:
    """
    Internal class.
    Provides some static methods that can be used to get information about available RAM and RAM usage on a device.
    """
    @staticmethod
    def get_total_ram_gb(device: str) -> float:
        device = str(device)
        if device == 'cpu':
            import psutil
            mem = psutil.virtual_memory()
            return mem.available / (1024. ** 3)
        elif device.startswith('cuda:'):
            gpu_num = int(device[5:])
            return GPUInfo.get_total_ram_gb(gpu_num=gpu_num)
        else:
            raise ValueError(f'Unknown device "{device}"')

    @staticmethod
    def get_process_ram_usage_gb(device: str, pid: Optional[int] = None) -> float:
        if pid is None:
            pid = os.getpid()  # get pid of this process
        device = str(device)
        if device == 'cpu':
            import psutil
            return psutil.Process(pid).memory_info().rss / 1024 ** 3
        elif device.startswith('cuda:'):
            gpu_num = int(device[5:])
            return GPUInfo.get_process_ram_usage_gb(gpu_num=gpu_num, pid=pid)
        else:
            raise ValueError(f'Unknown device "{device}"')


def measure_fixed_rams_gb(devices: List[str]) -> List[float]:
    """
    Internal method. Provides the RAM that is already used when using any PyTorch tensor on these devices.
    This can be significantly large due to the CUDA runtimes.
    :param devices: List of devices that should be used.
    :return: Returns a list of RAM usages (in GB) on these devices.
    """
    import torch
    # alloc dummy tensors to know how much memory PyTorch uses for its runtime
    dummy_tensors = [torch.ones(1).to(device) for device in devices]
    return [DeviceInfo.get_process_ram_usage_gb(device) for device in devices]


class FunctionRunner:
    """
    Simple helper class to run a function in a process / thread and submit the result back to a queue
    """
    def __init__(self, dill_f_and_args, result_queue):
        """
        :param dill_f_and_args: Should be a string that can be decoded with dill,
        i.e., dill_f_and_args = dill.dumps((f, args)) when we want to call f(*args) in the process.
        :param result_queue: Queue where the result should be pushed to.
        """
        self.dill_f_and_args = dill_f_and_args
        self.result_queue = result_queue

    def __call__(self):
        """
        Runs the function with the specified arguments and puts the result in the queue.
        """
        try:
            f, args = dill.loads(self.dill_f_and_args)
            result = f(*args)
            self.result_queue.put(result)
            self.result_queue.join()
        except Exception as e:
            print('Handling exception')
            print(e)
            traceback.print_exc()


class FunctionProcess:
    """
    Helper class to run a function in a separate process.
    """
    def __init__(self, f, *args):
        """
        :param f: Function to run in the process.
        :param args: Arguments to the function.
        """
        self.result_queue = mp.JoinableQueue()
        self.target = FunctionRunner(dill.dumps((f, args)), self.result_queue)
        self.process = mp.Process(target=self.target)

    def start(self) -> 'FunctionProcess':
        """
        Start the process computing the function.
        :return: Returns self such that we can use FunctionProcess(f, *args).start().
        """
        self.process.start()
        return self

    def is_done(self) -> bool:
        """
        :return: Returns true if the result of the function is available.
        """
        return not self.result_queue.empty()

    def get_ram_usage_gb(self, device='cpu') -> float:
        """
        :param device: PyTorch device string.
        :return: Returns the RAM usage in GB on the given device.
        """
        device = str(device)
        return DeviceInfo.get_process_ram_usage_gb(device, self.process.pid)

    def pop_result(self) -> Any:
        """
        :return: Returns the result and terminates the process, i.e., this function can only be called once.
        """
        result = self.result_queue.get()
        self.result_queue.task_done()
        self.process.terminate()
        self.process.join()
        return result


class AbstractJob:
    """
    Abstract base class for jobs that support execution and can return a RAM requirement and a name.
    """
    def __call__(self, device: str) -> None:
        """
        Abstract method to execute the job on the given device.
        :param device: PyTorch device to run the job on.
        """
        raise NotImplementedError()

    def get_ram_usage_gb(self) -> float:
        """
        Abstract method.
        :return: Should return the required RAM usage on the device in GB.
        Note that fixed RAM on the device due to PyTorch's CUDA runtime should not be included,
        as this is added automatically.
        """
        raise NotImplementedError()

    def get_desc(self) -> str:
        """
        Abstract method.
        :return: Should return a name that can be used for printing which job is running.
        """
        raise NotImplementedError()


class JobScheduler:
    """
    This class allows to run jobs with RAM constraints in separate processes on multiple devices in parallel.
    """
    _has_start_method_been_set = False

    def __init__(self, devices: Optional[List[str]] = None, use_gpu: bool = True, max_jobs_per_device: int = 1000):
        """
        :param devices: Optional list of PyTorch device strings.
        If None, the CPU is used if use_gpu=False and all available GPUs otherwise.
        :param use_gpu: Whether GPUs should be used if devices is None.
        :param max_jobs_per_device: Maximum number of jobs that are allowed to run per device.
        Note that the jobs also have RAM constraints,
        so the JobScheduler may run less jobs than the specified maximum number.
        """
        self.devices = (get_devices() if use_gpu else ['cpu']) if devices is None else devices
        self.max_jobs_per_device = max_jobs_per_device

    def run_all(self, jobs: List[AbstractJob]):
        """
        Run all jobs in separate processes, in parallel on multiple devices,
        according to the constraints imposed by the job's RAM bound and the maximum number of jobs per device.
        :param jobs: List of jobs.
        """
        if len(jobs) == 0:
            return
        if not JobScheduler._has_start_method_been_set:
            mp.set_start_method('spawn')
            JobScheduler._has_start_method_been_set = True
        start_time = time.time()

        print(f'Start time: {utils.format_date_s(start_time)}')

        max_ram_fraction = 0.9
        ram_gb_per_device = [DeviceInfo.get_total_ram_gb(device) * max_ram_fraction for device in self.devices]
        # execute this in a process such that reserved GPU memory can be released again
        fixed_ram_gb_per_device = FunctionProcess(measure_fixed_rams_gb, self.devices).start().pop_result()

        # started_infos: [{'process': ..., 'device': ..., 'ram_gb': ...}]
        started_infos = []
        # maybe start times?
        next_job_index = 0

        while next_job_index < len(jobs):
            next_job = jobs[next_job_index]
            job_ram_gb = next_job.get_ram_usage_gb()

            # try assigning a job to a device
            for i, device in enumerate(self.devices):
                n_jobs_on_device = len([si for si in started_infos if si['device'] == device])
                if n_jobs_on_device < self.max_jobs_per_device:
                    needs_ram_gb = job_ram_gb + fixed_ram_gb_per_device[i]
                    if needs_ram_gb > ram_gb_per_device[i]:
                        raise RuntimeError(f'RAM estimate of {needs_ram_gb:g} GB for job {next_job.get_desc()}'
                                           f' is too large for device')
                    remaining_ram = ram_gb_per_device[i] - sum([si['ram_gb'] for si in started_infos
                                                                if si['device'] == device], 0.0)
                    if needs_ram_gb < remaining_ram:
                        print(f'Starting job {next_job_index+1}/{len(jobs)} after {utils.format_length_s(time.time() - start_time)}')
                        started_infos.append({'process': FunctionProcess(next_job, device).start(),
                                              'device': device, 'ram_gb': needs_ram_gb})
                        next_job_index += 1
                        break
            else:  # no suitable device found
                # check which jobs are terminated
                new_terminated = False
                for i, si in enumerate(started_infos):
                    if si['process'].is_done():
                        si['process'].pop_result()
                        new_terminated = True
                        del started_infos[i]
                if not new_terminated:
                    time.sleep(0.1)

        # join all remaining processes
        for si in started_infos:
            si['process'].pop_result()

        end_time = time.time()
        print(f'End time: {utils.format_date_s(end_time)}')
        print(f'Total time: {utils.format_length_s(end_time - start_time)}')


class BatchALJob(AbstractJob):
    """
    This class implements the type of jobs used in Batch Active Learning, such that they can be used in JobScheduler.
    """
    def __init__(self, task: Task, split_id: int, trainer: ModelTrainer, ram_gb_per_sample: float, exp_name: str,
                 use_pool_for_normalization: bool, do_timing: bool, ram_gb_per_sample_bs: float):
        """
        :param task: Task to run.
        :param split_id: ID of the task split to run.
        :param trainer: ModelTrainer used for Batch Active Learning.
        :param ram_gb_per_sample: This should be an upper bound on how much RAM (in GB)
        per sample of the train+pool sets will be used. This might be on the order of 1e-5 or so.
        For details, see the implementation of get_ram_usage_gb().
        """
        self.task = task
        self.split_id = split_id
        self.trainer = trainer
        self.ram_gb_per_sample = ram_gb_per_sample
        self.ram_gb_per_sample_bs = ram_gb_per_sample_bs
        self.exp_name = exp_name
        self.use_pool_for_normalization = use_pool_for_normalization
        self.do_timing = do_timing

    def __call__(self, device: str):
        try:
            result_dict = self.trainer(TaskSplit(self.task, id=self.split_id,
                                                 use_pool_for_normalization=self.use_pool_for_normalization),
                                       device=device, do_timing=self.do_timing)
            utils.serialize(self.trainer.get_result_file_path(self.exp_name, self.task.task_name, self.split_id), result_dict,
                            use_json=True)
        except Exception as e:
            print(f'Error for alg {self.trainer.alg_name} on split {self.split_id} of task {self.task.task_name}:',
                  file=sys.stderr)
            print(e, file=sys.stderr)
            traceback.print_exc()
            print('', file=sys.stderr, flush=True)

    def get_ram_usage_gb(self) -> float:
        max_bs = (max(self.task.al_batch_sizes) if len(self.task.al_batch_sizes) > 0 else 0)
        return 0.2 + self.task.data_info.n_samples * self.task.data_info.n_features * 8 / (1024**3) \
               + (self.task.n_train + self.task.n_pool) * self.ram_gb_per_sample \
               + (self.task.n_train + self.task.n_pool) * max_bs * self.ram_gb_per_sample_bs

    def get_desc(self) -> str:
        return f'{self.trainer.alg_name} on split {self.split_id} of task {self.task.task_name}'


class JobRunner:
    """
    Simple class for adding Batch AL jobs and then running them in parallel using a JobScheduler.
    """
    def __init__(self, scheduler: JobScheduler):
        """
        :param scheduler: The JobScheduler configuration that should be used to run the jobs.
        """
        self.scheduler = scheduler
        self.jobs = []

    def add(self, exp_name: str, split_id: int, tasks: List[Task], ram_gb_per_sample: float,
            trainer: ModelTrainer, do_timing: bool, warn_if_exists: bool = True,
            use_pool_for_normalization: bool = True, ram_gb_per_sample_bs: float = 0.0):
        """
        Adds jobs for each task in tasks and for each split in range(self.n_splits)
        to the list of jobs that will be run in self.run_all().
        :param exp_name: experiment group name, is used for the top-level folder for saving
        :param split_id: id of the split to run on
        :param tasks: List of tasks that the trainer will be run on.
        :param ram_gb_per_sample: RAM estimate (in GB) per sample, typically of the order of 1e-5.
        An upper bound on the RAM usage (in GB) of a process is computed as
        0.2 + (size of data set in GB) + ram_gb_per_sample * (n_train + n_pool).
        :param trainer: ModelTrainer that runs a certain configuration of batch active learning.
        :param do_timing: Whether to take extra efforts (CUDA synchronization) for proper timing.
        :param warn_if_exists: Whether to print a message if the results already exist
        :param use_pool_for_normalization: whether to compute data normalization statistics also based on the pool data
        """
        for task in tasks:
            if utils.existsFile(trainer.get_result_file_path(exp_name, task.task_name, split_id)):
                if warn_if_exists:
                    print(f'Results already exist for {trainer.alg_name} on split {split_id} of task {task.task_name}',
                          flush=True)
                continue
            self.jobs.append(BatchALJob(task, split_id, trainer, ram_gb_per_sample, exp_name=exp_name,
                                        use_pool_for_normalization=use_pool_for_normalization,
                                        do_timing=do_timing, ram_gb_per_sample_bs=ram_gb_per_sample_bs))

    def run_all(self):
        """
        Runs all jobs on the job scheduler.
        """
        self.scheduler.run_all(self.jobs)




