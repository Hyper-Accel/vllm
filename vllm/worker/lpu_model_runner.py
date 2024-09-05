import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from hyperdex.transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
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
        print_logger(self.parallel_config.tensor_parallel_size)
        print_logger(num_device)
        #num_device = self.parallel_config.tensor_parallel_size
        print_logger(self.model_config.model)
        #NOTE(hyunjun): device number shoud be argumentize
        self.model = AutoModelForCausalLM.from_pretrained(hyperdex_ckpt, device_map={"gpu": 0, "lpu": num_device})
        self.tokenizer = AutoTokenizer.from_pretrained(hyperdex_ckpt)
        self.streamer = TextStreamer(self.tokenizer, use_print=False, use_sse=True, skip_special_tokens=True)

    def cleanup(self):
      del self.model
      del self.tokenizer
      del self.streamer

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
            #print_logger(seq_group_metadata.seq_data[0])
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
            # NOTE(hyunjun): Prompt is not currently supported by LPU
            # NOTE(woosuk): Since the FlashAttention kernel does not support
            # ragged inputs, we split the prompts into different batches and
            # process them separately. This is a temporary hack that should be
            # optimized by using SplashAttention.
#            next_token_ids = []
#            orig_slot_mapping = model_input.attn_metadata.slot_mapping
#            batch_size = model_input.input_lens.shape[0]
#            start_idx = 0
#            for i in range(batch_size):
#                # Get the actual prefill_len.
#                prefill_len = model_input.input_lens[i:i + 1].item()
#                prefill_len = _get_padded_prefill_len(prefill_len)
#                end_idx = start_idx + prefill_len
#
#                model_input.attn_metadata.slot_mapping = orig_slot_mapping[
#                    None, start_idx:end_idx]
#                model_input.attn_metadata.num_prefills = 1
#                output_token_ids = _execute_model(
#                    model_input.token_ids[None, start_idx:end_idx],
#                    model_input.position_ids[None, start_idx:end_idx],
#                    model_input.attn_metadata,
#                    model_input.input_lens[i:i + 1],
#                    model_input.t[i:i + 1],
#                    model_input.p[i:i + 1],
#                    model_input.num_samples,
#                    kv_caches,
#                    clone=True)
#                # Retrieve the outputs to CPU.
#                next_token_ids += output_token_ids.cpu().tolist()
#                start_idx = end_idx
        else:
            # Execute the model.
    #        output_token_ids = _execute_model(
    #            model_input.token_ids, model_input.position_ids,
    #            model_input.attn_metadata, model_input.input_lens,
    #            model_input.t, model_input.p, model_input.num_samples,
    #            kv_caches)
            print_logger(model_input.token_ids)
            print_logger(model_input.max_tokens)
            if self.model_execution == False:
              import time
              t1 = time.time()
              print_logger(model_input)
              #print_logger(model_input.temperature)
              #print_logger(model_input.repetition_penalty)
              print_logger(model_input.token_ids)
              for i in range(len(model_input.token_ids)):
               print(model_input.max_tokens[i], model_input.p[i], model_input.k[i], model_input.t[i], model_input.r[i])
               output_token_ids = self.model.generate(
                model_input.token_ids[i].tolist(),
                max_new_tokens=model_input.max_tokens[i],
                do_sample=True,
                top_p=model_input.p[i],
                top_k=model_input.k[i],
                temperature=model_input.t[i],
                #repetition_penalty=model_input.r[i]
                )
               self.output_token_ids_buffer = output_token_ids[len(model_input.token_ids[i]):] 
              #TODO: should be modified to support batch 
              t2 = time.time()
              #self.cleanup() #TODO: it should exist in llm_engine
              print("Core Computation Latency : ", str(t2-t1))
              self.model_execution = True
              from transformers import AutoTokenizer
              tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
              tmp_output = tokenizer.decode(self.output_token_ids_buffer)
              print_logger(tmp_output)
            
            #output_token_ids = output_token_ids[len(model_input.token_ids):]
            #print_logger(output_token_ids) 
            # Retrieve the outputs to CPU.
            next_token_id = self.output_token_ids_buffer[self.iteration] #.cpu().tolist()
            self.iteration = self.iteration + 1
            print_logger(self.iteration)
            print_logger(model_input.max_tokens[0])
            print_logger(type(model_input.max_tokens[0]))
            print_logger(len(self.output_token_ids_buffer))
            print_logger(type(len(self.output_token_ids_buffer)))
            print_logger(len(self.output_token_ids_buffer) == self.iteration) # Add Stop Token Position
            if len(self.output_token_ids_buffer) == self.iteration:
              print_logger("After")
              self.model_execution = False
              self.iteration = 0
   
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
