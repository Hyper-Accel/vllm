import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

import transformers
#  from transformers import AutoModelForCausalLM, AutoTokenizer
# HyperDex package
from hyperdex.transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from hyperdex.tools.huggingface import AutoModelConverter
from hyperdex.tools.mem_map import memory_mapper
from hyperdex.tools.inst_gen import inst_generator

#import torch_xla.core.xla_model as xm
#import torch_xla.runtime as xr

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         MultiModalConfig, ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger, print_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SamplerOutput, SequenceGroupMetadata,
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
        for company in os.listdir("/opt/hyperdex/models"):
            for model_name in os.listdir(os.path.join("/opt/hyperdex/models", company)):
                model_list.append(os.path.join(company, model_name))
        # Get the model checkpoint path
        model_path = os.path.join("/opt/hyperdex/models", model_id)
        model_ckpt = os.path.join(model_path, "ckpt")
        # Download the model if it does not exist
        if model_id not in model_list:
        # Initialize HuggingFace access token
            hf_access_token = None
            while True:
                try:
                    print("[Info\t] Download model at {}".format(model_ckpt))
                    transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=hf_access_token).save_pretrained(model_ckpt)
                    transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_access_token).save_pretrained(model_ckpt)
                except:
                    if hf_access_token is None:
                        print("[Info\t] Repo model {} is gated. You must be authenticated to access it.".format(model_id))
                        hf_access_token = input("\033[0m[\033[32mOption\t\033[0m] Please enter the HuggingFace access token: ")
                        continue
                    else:
                        print("\033[0m[\033[31mError\033[0m\t] \"{}\" model does not exist in HuggingFace Hub".format(model_id))
                        raise RuntimeError("Please check the huggingface model id")
                break
            # Use pre-downloaded model
        else:
            print("[Info\t] Found model at {}".format(model_ckpt))
            print("[Info\t] Skip download")

    # Convert model
    def convert(model_ckpt: str):
        print("[Info\t] Convert the model to HyperDex model format")
        # Check if the checkpoint exist
        bin_exist = os.path.isfile(os.path.join(model_ckpt, "hyperdex_model.bin"))
        cfg_exist = os.path.isfile(os.path.join(model_ckpt, "hyperdex_config.json"))
        if bin_exist and cfg_exist:
            print("[Info\t] Found binary at {}".format(os.path.join(model_ckpt, "hyperdex_model.bin")))
            print("[Info\t] Found config at ls{}".format(os.path.join(model_ckpt, "hyperdex_config.json")))
            print("[Info\t] Skip converting")
        else:
            hf_converter = AutoModelConverter(model_ckpt)
            hf_converter.convert(model_ckpt)
            hf_converter.save(model_ckpt)
            print("[Info\t] Save the converted checkpoint at {}".format(model_ckpt))

    # Mapping model and i/o
    def mapping(model_path: str, num_device: int):
        # Check if the number of device is invalid
        if num_device > 0 and (num_device & (num_device - 1)) == 0:
            print("[Info\t] Optimize the model paramater")
            mapper = memory_mapper(model=model_path, num_device=num_device, slib="/opt/hyperdex/compiler/lib/libhdexmm.so")

            ddr_config = "config_" + str(num_device) + "fpga_ddr.ini"
            io_exist = os.path.isfile(os.path.join(model_path, "config", ddr_config))
            if io_exist:
                print("[Info\t] Found io at ", os.path.join(model_path, "config"))
                print("[Info\t] Skip io memory mapping")
            else:
                mapper.io()
            
            model_param = os.listdir(os.path.join(model_path, "param"))
            hbm_config = "config_" + str(num_device) + "fpga_hbm.ini"
            prm_exist = len([param for param in model_param if
                             f"param_{str(num_device)}" in param]) == 32 * num_device and os.path.isfile(
                                 os.path.join(model_path, "config", hbm_config)
                             )
            if prm_exist:
                print("[Info\t] Found param at ", model_path)
                print("[Info\t] Skip parameter memory mapping")
            else:
                mapper.param()
            print("[Info\t] Save the optimized data at {}/param".format(model_path))
        else:
            print("\033[0m[\033[31mError\033[0m\t] The number of devices should be a power of two. (ex. 1, 2, 4, 8, etc.)")
            raise RuntimeError("The number of devices is not a power of two!")

    # Generate instruction
    def instruction(model_id, num_device):
        print("[Info\t] Optimize the model instruction")
        generator = inst_generator(model=model_id, prefix="/opt/hyperdex/models", num_device=num_device, slib="/opt/hyperdex/compiler/lib/libhdexig.so")
        model_inst = os.listdir(os.path.join("/opt/hyperdex/models/", model_id, "inst"))
        inst_exist = len([inst for inst in model_inst if f"_{str(num_device)}fpga.bin" in inst]) == 2
        if inst_exist:
            print("[Info\t] Found instructions at ", os.path.join("/opt/hyperdex/models/", model_id, "inst"))
            print("[Info\t] Skip instructions memory mapping")
        else:
            generator.compile()
        print("[Info\t] Save the optimized instruction at /opt/hyperdex/models/{}/inst".format(model_id))

@dataclass(frozen=True)
class ModelInputForLPU(ModelRunnerInputBase):
    token_ids: torch.Tensor
    position_ids: torch.Tensor 
    #attn_metadata: AttentionMetadata
    #attn_metadata: Optional[AttentionMetadata] = None,
    input_lens: torch.Tensor
    t: List[int]
    p: List[int]
    k: List[int]
    r: List[int]
    max_tokens: List[int]
    min_tokens: List[int]
#    num_samples: int
#    best_of: List[int]
    seq_groups: List[List[int]]
#    virtual_engine: int = 0

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "token_ids": self.token_ids,
            "position_ids": self.position_ids,
            "input_lens": self.input_lens,
            "t": self.t,
            "p": self.p,
            "k": self.p,
            "r": self.r,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
#            "num_samples": self.num_samples,
#            "best_of": self.best_of,
            "seq_groups": self.seq_groups,
#            "virtual_engine": self.virtual_engine,
        }
        #_add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
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
        #self.multimodal_config = multimodal_config
        self.is_driver_worker = is_driver_worker
        #self.output_token_ids = None

        #self.block_size = self.cache_config.block_size
        #self.max_num_blocks_per_seq = (self.model_config.max_model_len //
        #                               self.block_size)
        #self.block_tables = np.zeros(
        #    (self.scheduler_config.max_num_seqs, self.max_num_blocks_per_seq),
        #    dtype=np.int32)
        #self.attn_backend = get_attn_backend(
        #    self.model_config.get_num_attention_heads(self.parallel_config),
        #    self.model_config.get_head_size(),
        #    self.model_config.get_num_kv_heads(self.parallel_config),
        #    self.model_config.get_sliding_window(),
        #    self.model_config.dtype,
        #    self.cache_config.cache_dtype,
        #    self.block_size,
        #    False,
        #)
        # LPU custom attribute
        self.model_execution = False
        self.output_token_ids_buffer = []
        self.iteration = 0


    def load_model(self, num_device = 1) -> None:
        hyperdex_ckpt = "/opt/hyperdex/models/" + self.model_config.model
        HyperDexSDK.compile(HyperDexSDK, hyperdex_ckpt, num_device)
        #NOTE(hyunjun): device number shoud be argumentize
        self.model = AutoModelForCausalLM.from_pretrained(hyperdex_ckpt, device_map={"gpu": 0, "lpu": num_device})
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
        return input_tokens, input_positions, prompt_lens #attn_metadata

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor]: #, AttentionMetadata, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []

        batch_idx = 0
        for seq_group_metadata in seq_group_metadata_list:
            #assert not seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            print_logger(seq_ids)
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                #input_tokens.append([generation_token])
                input_tokens.append(list(seq_data.prompt_token_ids))

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

        #input_tokens = seq_group_metadata.seq_data.get_prompt_token_ids
        batch_size = batch_idx #_get_padded_batch_size(batch_idx)  #HJ: Currently, LPU cannot support batching
        num_paddings = batch_size - batch_idx
        input_tokens = input_tokens + [[0]] * num_paddings
        print_logger(input_tokens)
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
        return input_tokens, input_positions, input_lens #attn_metadata

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
#        padded_batch_size: int,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
        assert len(seq_group_metadata_list) > 0 # t p k r max min
        t = []
        p = []
        k = []
        r = []
        max_tokens = []
        min_tokens = []
        for seq_group_metadata in seq_group_metadata_list:
            sampling_params = seq_group_metadata.sampling_params
            t.append(sampling_params.temperature)
            p.append(sampling_params.top_p)
            k.append(sampling_params.top_k)
            r.append(sampling_params.repetition_penalty)
            max_tokens.append(sampling_params.max_tokens)
            min_tokens.append(sampling_params.min_tokens)
            if sampling_params.use_beam_search:
                raise NotImplementedError(
                    "Beam search is not supported by the LPU backend.")
            if sampling_params.logprobs is not None:
                raise NotImplementedError(
                    "logprobs is not currently supported by the LPU backend.")
            if sampling_params.prompt_logprobs is not None:
                raise NotImplementedError(
                    "prompt_logprobs is not currently supported by the LPU "
                    "backend.")

        return t, p, k, r, max_tokens, min_tokens

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForLPU:
        del finished_requests_ids  # Unused.
        #assert virtual_engine == 0
        assert len(seq_group_metadata_list) > 0
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        print_logger(seq_group_metadata_list[0])
        is_prompt = False
        if is_prompt:
            inputs = self._prepare_prompt(seq_group_metadata_list)
        else:
            inputs = self._prepare_decode(seq_group_metadata_list)
        input_tokens, input_positions, input_lens = inputs #attn_metadata # [5] [1]
#        padded_batch_size = input_tokens.shape[0] # input token shape issue
        print_logger(input_tokens, 3)
        t, p, k, r, max_tokens, min_tokens = self._prepare_sample(seq_group_metadata_list)
#                                             padded_batch_size)
#        num_samples = _MAX_NUM_SAMPLES if is_prompt else 1

        seq_groups = [
            list(metadata.seq_data.keys())
            for metadata in seq_group_metadata_list
        ]
        return ModelInputForLPU(input_tokens, input_positions, #None, #attn_metadata,
                                input_lens, t, p, k, r, max_tokens, min_tokens, #num_samples, best_of,
                                seq_groups)

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForLPU:
        model_input = ModelInputForLPU.from_broadcasted_tensor_dict(
            tensor_dict) #, attn_backend=self.attn_backend)
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
        print_logger(model_input)

        num_prefills = 0
        is_prompt = num_prefills > 0
        if is_prompt:
            print_logger(is_prompt)
        else:
            # Execute the model.
            print_logger(model_input.token_ids)
            print_logger(model_input.max_tokens)
            if self.model_execution == False:
              #import time
              #t1 = time.time()
              for i in range(len(model_input.token_ids)):
               print(model_input.max_tokens[i], model_input.p[i], model_input.k[i], model_input.t[i], model_input.r[i])
               self.output_token_ids = self.model.generate_yield(
                model_input.token_ids[i].tolist(),
                max_new_tokens=model_input.max_tokens[i],
                do_sample=True,
                top_p=model_input.p[i],
                top_k=model_input.k[i],
                temperature=model_input.t[i],
                streamer=self.streamer
                #repetition_penalty=model_input.r[i]
                )
               #self.output_token_ids_buffer = output_token_ids[len(model_input.token_ids[i]):] 
              #TODO: should be modified to support batch 
              #t2 = time.time()
              #print("Core Computation Latency : ", str(t2-t1))
              self.model_execution = True
              #from transformers import AutoTokenizer
              #tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
              #tmp_output = tokenizer.decode(self.output_token_ids_buffer)
              #print_logger(tmp_output)
            
            #output_token_ids = output_token_ids[len(model_input.token_ids):]
            #print_logger(output_token_ids) 
            # Retrieve the outputs to CPU.
            print_logger(self.iteration)
            print_logger(model_input.max_tokens[0])
            #for next_token in self.output_token_ids:
            #self.next_token = next(self.output_token_ids)
            #next_token_id = self.next_token[0]
            #next_token_id = self.output_token_ids_buffer[self.iteration] #.cpu().tolist()
            next_token_id = next(self.output_token_ids)[0]
            print_logger(next_token_id)
            self.iteration = self.iteration + 1
            if (model_input.max_tokens[0]) == self.iteration:
              print_logger("After")
              self.model_execution = False
              self.iteration = 0
              #del self.next_token
              #del self.output_token_ids
              #self.output_token_ids = None
   
            # NOTE(woosuk): Minimal code to construct the sampler outputs.
            # The LPU backend does not reuse the sampler, since the LPU backend
            # does not support the advanced sampling parameters such as logprobs.
            print_logger(next_token_id)
            zero_logprob = Logprob(0.0)
            batch_idx = 0
            sampler_outputs = []
            for seq_group in model_input.seq_groups:
                seq_ids = seq_group
                print_logger(model_input)
                print_logger(seq_group)
                seq_outputs = []
                if is_prompt:
                    print_logger(is_prompt)
                    #NOTE(hyunjun) Prompt is not currently supported by LPU
                    #assert len(seq_ids) == 1
                    #seq_id = seq_ids[0]
                    #for i in range(model_input.best_of[batch_idx]):
                    #    next_token_id = next_token_ids[batch_idx][i]
                    #    seq_outputs.append(
                    #        SequenceOutput(seq_id, next_token_id,
                    #                       {next_token_id: zero_logprob}))
                    #batch_idx += 1
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
