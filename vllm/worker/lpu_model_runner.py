import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Any, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

import transformers

# HyperDex package
from hyperdex.transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from hyperdex.tools.huggingface import AutoModelConverter
from hyperdex.tools.mem_map import memory_mapper
from hyperdex.tools.inst_gen import inst_generator

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         MultiModalConfig, ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger, print_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceGroupMetadata,
                           SequenceOutput)
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase,
    _add_attn_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME(woosuk): Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000
# FIXME(woosuk): Temporarily disabled top-p sampling since it's too slow.
_ENABLE_TOP_P = False
# FIXME(woosuk): A temporary hack to support `n > 1`.
# This can significantly affect the performance if too large.
_MAX_NUM_SAMPLES = 128

# HyperDex Path
MODEL_PATH = "/opt/hyperdex/models"
COMPILER_PATH = "/opt/hyperdex/compiler/lib"

class HyperDexSDK:
    def compile(self, model_ckpt, num_device):
        model_id = model_ckpt.split("/")[-2] + "/" + model_ckpt.split("/")[-1]
        self.download(model_id)
        self.convert(os.path.join(model_ckpt, "ckpt"))
        self.mapping(model_ckpt, num_device)
        self.instruction(model_id, num_device)

    # Download model
    def download(model_id: str):
        # Get the installed model list
        model_list = []
        for company in os.listdir(MODEL_PATH):
            for model_name in os.listdir(os.path.join(MODEL_PATH, company)):
                model_list.append(os.path.join(company, model_name))
        # Get the model checkpoint path
        model_path = os.path.join(MODEL_PATH, model_id)
        model_ckpt = os.path.join(model_path, "ckpt")
        # Download the model if it does not exist
        if model_id not in model_list:
        # Initialize HuggingFace access token
            hf_access_token = None
            while True:
                try:
                    print("[INFO\t] Download model at {}".format(model_ckpt))
                    transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=hf_access_token).save_pretrained(model_ckpt)
                    transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_access_token).save_pretrained(model_ckpt)
                except:
                    if hf_access_token is None:
                        print("[INFO\t] Repo model {} is gated. You must be authenticated to access it.".format(model_id))
                        hf_access_token = input("\033[0m[\033[32mOption\t\033[0m] Please enter the HuggingFace access token: ")
                        pass
                    else:
                        print("\033[0m[\033[31mError\033[0m\t] \"{}\" model does not exist in HuggingFace Hub".format(model_id))
                        raise RuntimeError("Please check the huggingface model id")
                break
            # Use pre-downloaded model
        else:
            pass
            #print("[INFO\t] Found model at {}".format(model_ckpt))
            #print("[INFO\t] Skip download")

    # Convert model
    def convert(model_ckpt: str):
        #print("[INFO\t] Convert the model to HyperDex model format")
        # Check if the checkpoint exist
        bin_exist = os.path.isfile(os.path.join(model_ckpt, "hyperdex_model.bin"))
        cfg_exist = os.path.isfile(os.path.join(model_ckpt, "hyperdex_config.json"))
        if bin_exist and cfg_exist:
            #print("[INFO\t] Found binary at {}".format(os.path.join(model_ckpt, "hyperdex_model.bin")))
            #print("[INFO\t] Found config at ls{}".format(os.path.join(model_ckpt, "hyperdex_config.json")))
            #print("[INFO\t] Skip converting")
            pass
        else:
            hf_converter = AutoModelConverter(model_ckpt)
            hf_converter.convert(model_ckpt)
            hf_converter.save(model_ckpt)
            print("[INFO\t] Save the converted checkpoint at {}".format(model_ckpt))

    # Mapping model and i/o
    def mapping(model_path: str, num_device: int):
        # Check if the number of device is invalid
        if num_device > 0 and (num_device & (num_device - 1)) == 0:
            #print("[INFO\t] Optimize the model paramater")
            mapper = memory_mapper(model=model_path, num_device=num_device, slib=os.path.join(COMPILER_PATH, "libhdexmm.so"))

            ddr_config = "config_" + str(num_device) + "fpga_ddr.ini"
            io_exist = os.path.isfile(os.path.join(model_path, "config", ddr_config))
            if io_exist:
                #print("[INFO\t] Found io at ", os.path.join(model_path, "config"))
                #print("[INFO\t] Skip io memory mapping")
                pass
            else:
                mapper.io()
            
            model_param = os.listdir(os.path.join(model_path, "param"))
            hbm_config = "config_" + str(num_device) + "fpga_hbm.ini"
            prm_exist = len([param for param in model_param if
                             f"param_{str(num_device)}" in param]) == 32 * num_device and os.path.isfile(
                                 os.path.join(model_path, "config", hbm_config)
                             )
            if prm_exist:
                pass
                #print("[INFO\t] Found param at ", model_path)
                #print("[INFO\t] Skip parameter memory mapping")
            else:
                mapper.param()
                print("[INFO\t] Save the optimized data at {}/param".format(model_path))
        else:
            print("\033[0m[\033[31mError\033[0m\t] The number of devices should be a power of two. (ex. 1, 2, 4, 8, etc.)")
            raise RuntimeError("The number of devices is not a power of two!")

    # Generate instruction
    def instruction(model_id, num_device):
        #print("[INFO\t] Optimize the model instruction")
        generator = inst_generator(model=model_id, prefix=MODEL_PATH, num_device=num_device, slib=os.path.join(COMPILER_PATH, "libhdexig.so"))
        model_inst = os.listdir(os.path.join(MODEL_PATH, model_id, "inst"))
        inst_exist = len([inst for inst in model_inst if f"_{str(num_device)}fpga.bin" in inst]) == 2
        if inst_exist:
            pass
            #print("[INFO\t] Found instructions at ", os.path.join(MODEL_PATH, model_id, "inst"))
            #print("[INFO\t] Skip instructions memory mapping")
        else:
            generator.compile()
            print("[INFO\t] Save the optimized instruction at /opt/hyperdex/models/{}/inst".format(model_id))


@dataclass(frozen=True)
class ModelInputForLPU(ModelRunnerInputBase):
    # NOTE (hyunjun): sampling param variables are list type to support batching later
    token_ids: torch.Tensor
    position_ids: torch.Tensor 
    input_lens: torch.Tensor
    t: List[int]
    p: List[int]
    k: List[int]
    r: List[int]
    stop: List[int]
    max_tokens: List[int]
    min_tokens: List[int]
    seq_groups: List[List[int]]

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "token_ids": self.token_ids,
            "position_ids": self.position_ids,
            "input_lens": self.input_lens,
            "t": self.t,
            "p": self.p,
            "k": self.k,
            "r": self.r,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "seq_groups": self.seq_groups,
        }
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type["ModelInputForLPU"],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForLPU":
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class LPUModelRunner(ModelRunnerBase[ModelInputForLPU]):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        multimodal_config: Optional[MultiModalConfig] = None,
        is_driver_worker: bool = False,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        # LPU custom attribute
        self.model_execution = False
        self.output_token_ids_buffer = []
        self.iteration = 0


    def load_model(self, num_gpu_devices = 0, num_lpu_devices = 1) -> None:
        hyperdex_ckpt = os.path.join(MODEL_PATH, self.model_config.model)
        HyperDexSDK.compile(HyperDexSDK, hyperdex_ckpt, num_lpu_devices)
        # NOTE(hyunjun): The number of GPU should be added
        #print_logger(num_gpu_devices)
        #print_logger(num_lpu_devices)
        self.model = AutoModelForCausalLM.from_pretrained(hyperdex_ckpt, device_map={"gpu": num_gpu_devices, "lpu": num_lpu_devices})
        self.tokenizer = AutoTokenizer.from_pretrained(hyperdex_ckpt)
        self.streamer = TextStreamer(self.tokenizer, use_print=False, use_sse=True, skip_special_tokens=True)
        self.output_token_ids = None

    def cleanup(self):
      del self.model
      del self.tokenizer
      del self.streamer
      del self.output_token_ids

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        prompt_lens: List[int] = []
        slot_mapping: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            # Could include output tokens when a request is preempted.
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            input_positions.extend(list(range(prompt_len)))

            padded_prompt_len = _get_padded_prefill_len(prompt_len)
            num_paddings = padded_prompt_len - prompt_len
            input_tokens += [0] * num_paddings
            input_positions += [0] * num_paddings

        assert len(prompt_lens) > 0
        num_prefills = len(prompt_lens)
        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.int32,
                                    device="cpu")
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.int32,
                                       device="cpu")
        prompt_lens = torch.tensor(prompt_lens,
                                   dtype=torch.int32,
                                   device="cpu")
        return input_tokens, input_positions, prompt_lens 

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []

        batch_idx = 0
        for seq_group_metadata in seq_group_metadata_list:
            #assert not seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                #input_tokens.append([generation_token])
                input_tokens.append(list(seq_data.prompt_token_ids))

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

        batch_size = batch_idx 
        # NOTE(hyunjun): Currently, LPU cannot support batching. So there is no padding 
        num_paddings = batch_size - batch_idx
        input_tokens = input_tokens + [[0]] * num_paddings
        input_positions = input_positions + [[0]] * num_paddings

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.int32,
                                    device="cpu")
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.int32,
                                       device="cpu")
        input_lens = torch.tensor([1] * batch_size,
                                  dtype=torch.int32,
                                  device="cpu")
        return input_tokens, input_positions, input_lens 

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
        assert len(seq_group_metadata_list) > 0 
        t = []
        p = []
        k = []
        r = []
        stop = []
        max_tokens = []
        min_tokens = []
        for seq_group_metadata in seq_group_metadata_list:
            sampling_params = seq_group_metadata.sampling_params
            t.append(sampling_params.temperature)
            p.append(sampling_params.top_p)
            k.append(sampling_params.top_k)
            r.append(sampling_params.repetition_penalty)
            stop.append(sampling_params.stop_token_ids)
            max_tokens.append(sampling_params.max_tokens)
            min_tokens.append(sampling_params.min_tokens)
            if sampling_params.logprobs is not None:
                raise NotImplementedError(
                    "logprobs is not currently supported by the LPU backend.")
            if sampling_params.prompt_logprobs is not None:
                raise NotImplementedError(
                    "prompt_logprobs is not currently supported by the LPU "
                    "backend.")

        return t, p, k, r, stop, max_tokens, min_tokens

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForLPU:
        del finished_requests_ids  # Unused.
        #assert virtual_engine == 0
        assert len(seq_group_metadata_list) > 0
        # NOTE: We assume that all sequences in the group are all prompts or all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        is_prompt = False
        if is_prompt:
            inputs = self._prepare_prompt(seq_group_metadata_list)
        else:
            inputs = self._prepare_decode(seq_group_metadata_list)
        input_tokens, input_positions, input_lens = inputs
        t, p, k, r, stop, max_tokens, min_tokens = self._prepare_sample(seq_group_metadata_list)

        seq_groups = [
            list(metadata.seq_data.keys())
            for metadata in seq_group_metadata_list
        ]
        return ModelInputForLPU(input_tokens, input_positions, input_lens,
                                t, p, k, r, stop, max_tokens, min_tokens, seq_groups)

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForLPU:
        model_input = ModelInputForLPU.from_broadcasted_tensor_dict(
            tensor_dict) 
        return model_input

    @torch.no_grad()
    def execute_model(
        self,
        model_input: ModelInputForLPU,
        kv_caches: Optional[List[Any]],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> List[SamplerOutput]:
        assert intermediate_tensors is None
        if num_steps > 1:
            raise ValueError(
                "LPUModelRunner does not support multi-step execution.")

        num_prefills = 0
        is_prompt = num_prefills > 0
        if is_prompt:
            #NOTE(hyunjun) Prompt is not currently supported by LPU
            pass
        else:
            # Execute the model.
            if self.model_execution == False:
              # NOTE(hyunjun): Currently len(model_input.token_ids) is zero because LPU does not support batch
              for i in range(len(model_input.token_ids)):
               self.output_token_ids = self.model.generate_yield(
                model_input.token_ids[i].tolist(),
                max_new_tokens=model_input.max_tokens[i],
                do_sample=True,
                top_p=model_input.p[i],
                top_k=model_input.k[i],
                temperature=model_input.t[i],
                repetition_penalty=model_input.r[i],
                stop=model_input.stop[i],
                streamer=self.streamer
                )
              # NOTE(hyunjun): LPU will support batching later, but not now
              self.model_execution = True
            
            # NOTE(hyunjun): To execute vllm serve, hyperdex/transformers/backend/lpu/transformers should be modified (streamer.end)
            next_token_id = next(self.output_token_ids)[0]
            self.iteration = self.iteration + 1
            if (model_input.max_tokens[0]) == self.iteration:
              self.model_execution = False
              self.iteration = 0
   
            # NOTE(woosuk): Minimal code to construct the sampler outputs.
            # The LPU backend does not reuse the sampler, since the LPU backend
            # does not support the advanced sampling parameters such as logprobs.
            zero_logprob = Logprob(0.0)
            batch_idx = 0
            sampler_outputs = []
            for seq_group in model_input.seq_groups:
                seq_ids = seq_group
                seq_outputs = []
                if is_prompt:
                    # NOTE(hyunjun) Prompt is not currently supported by LPU
                    pass
                else:
                    for seq_id in seq_ids:
                        seq_outputs.append(
                            SequenceOutput(seq_id, next_token_id,
                                           {next_token_id: zero_logprob}))
                        batch_idx += 1
                    sampler_outputs.append(
                        CompletionSequenceGroupOutput(seq_outputs, None))
            return [SamplerOutput(sampler_outputs)]


def _get_padded_prefill_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()


def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16


def _apply_top_p(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    logits_sorted = torch.sort(logits, dim=-1, descending=True).values
    sorted_cum_probs = torch.cumsum(logits_sorted.softmax(dim=-1), dim=-1)
    cutoff_index = torch.sum(sorted_cum_probs < p, dim=-1, keepdim=True)
    cutoff_logit = torch.gather(logits_sorted, -1, cutoff_index)
    logits = logits.masked_fill_(logits < cutoff_logit, -float("inf"))
    return logits
