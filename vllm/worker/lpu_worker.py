import os
from typing import List, Optional, Tuple, Union

import torch
from hyperdex.transformers import AutoModelForCausalLM, AutoTokenizer
#import torch_xla.core.xla_model as xm
#import torch_xla.runtime as xr

import vllm.envs as envs
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         MultiModalConfig, ParallelConfig, SchedulerConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger, print_logger
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.worker.lpu_model_runner import LPUModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerInput)

logger = init_logger(__name__)


class LPUWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        #multimodal_config: Optional[MultiModalConfig],
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.parallel_config.rank = rank
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        #self.multimodal_config = multimodal_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        assert self.device_config.device_type == "fpga"
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        self.model_runner: LPUModelRunner = LPUModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config,
            #multimodal_config,
            is_driver_worker=is_driver_worker)
    
    def cleanup(self):
      self.model_runner.cleanup()

    def init_device(self) -> None:
        self.device = torch.device("fpga")
        self.device_config.device = self.device
        print_logger("Hello")
        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )
        print_logger("Hello")
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size)
        print_logger("Hello")
 
#         # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    # LPU does not support this function
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        return 1, 0 

    # LPU does not support this function
    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        pass 

    # LPU does not support this function
    def _warmup_model(self) -> None:
        pass 

    # LPU does not support this function
    def get_cache_block_size_bytes(self) -> int:
        pass 
        return 1024

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    # LPU does not support this function
    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return None 

    # LPU does not support this function
    def prepare_worker_input(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> WorkerInput:
        print_logger(execute_model_req)
        virtual_engine = execute_model_req.virtual_engine
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        blocks_to_swap_in = _make_src_to_dst(
            execute_model_req.blocks_to_swap_in, "cpu", "cpu")              # self.device
        blocks_to_swap_out = _make_src_to_dst(
            execute_model_req.blocks_to_swap_out, "cpu", "cpu") # self.device
        blocks_to_copy = _make_src_to_dst(execute_model_req.blocks_to_copy,
                                          "cpu", "cpu")
        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
        )

    # LPU does not support this function
    def execute_worker(self, worker_input: WorkerInput) -> None:
      pass


def _make_src_to_dst(
    mapping: List[Tuple[int, int]],
    src_device: Union[torch.device, str],
    dst_device: Union[torch.device, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    src_indices = [i for i, _ in mapping]
    dst_indices = [i for _, i in mapping]
    src_indices = torch.tensor(src_indices,
                               device=src_device,
                               dtype=torch.int64)
    dst_indices = torch.tensor(dst_indices,
                               device=dst_device,
                               dtype=torch.int64)
    return src_indices, dst_indices


@torch.compile(backend="openxla")
def _insert_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    lpu_k_cache: torch.Tensor,
    lpu_v_cache: torch.Tensor,
) -> None:
    torch.ops.xla.dynamo_set_buffer_donor_(lpu_k_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(lpu_v_cache, True)
    lpu_k_cache[:, indices] = k
    lpu_v_cache[:, indices] = v
