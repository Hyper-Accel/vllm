##########################################################################
##  Developer Tool Set for HyperDex Runtime Library
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.0
##  Date:     2023-03-01      ( v1.0.0  mobility demo       )
##            2023-03-29      ( v1.1.0, code refactoring    )
##            2023-12-07      ( v1.3.0, migrate to rt-2     )
##
##########################################################################

import os
import random
import json
import numpy as np
import torch

# HyperDex Runtime Library
import hrt

##########################################################################
## Defines
##########################################################################

NETWORK_TABLE_PREFIX = "/opt/hyperdex/hrt/xclbin"

##########################################################################
## Class
##########################################################################

class TextStreamer:
  
  # Constructor
  def __init__(self, tokenizer, use_print=True, use_sse=False, **decode_kwargs):
    self.tokenizer = tokenizer
    self.use_print = use_print
    self.use_sse = use_sse
    self.decode_kwargs = decode_kwargs
    # Variables used in the streaming process
    self.position = 0
    self.token_cache = []
    self.print_len = 0
    # Vairables used in sse
    self.all_input_ids = None
    self.prev_tokens = None
    # Warming up the tokenizer
    self.tokenizer.decode([0])

  # Check the print mode
  def is_using_print(self):
    return self.use_print

  # Check the sse mode
  def is_using_sse(self):
    return self.use_sse

  # Check the special tokens:
  def is_special_token(self, text):
    return text in self.tokenizer.special_tokens_map.values()

  ''' Text Streaming '''
  # Get the valid output token by checking buffer
  def get(self, buffer, skip):  
    # Get the valid token with infinite loop
    while True:
      token = self.read_single_token(buffer, skip)
      if token != 0xffffffff:
        self.position = self.position + 1
        return token

  # Receives tokens, decodes them, and prints them to stdout as soon as they form entire words
  def put(self, token):
    # Add the new token to the cache and decodes the entire thing
    self.token_cache.append(token)
    text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
    
    # After the symbol for a new line, we flush the cache
    if text.endswith("\n"):
      printable_text = text[self.print_len :]
      self.token_cache = []
      self.print_len = 0
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
      printable_text = text[self.print_len : text.rfind(" ") + 1]
      self.print_len += len(printable_text)
    
    # Return printable text
    if self.use_print:
      self.print_finalized_text(printable_text)
    else:
      return printable_text

  # Flushes any remaining cache and prints a newline to stdout
  def end(self):
    # Flush position only for SSE mode
    if self.use_sse:
      self.position = 0
    # Flush the cache, if it exists
    elif len(self.token_cache) > 0:
      text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
      printable_text = text[self.print_len :]
      self.position = 0
      self.token_cache = []
      self.print_len = 0
    else:
      self.position = 0
      printable_text = ""
      
    # Return printable text
    if self.use_sse:
      pass
    elif self.use_print:
      self.print_finalized_text(printable_text, stream_end=True)
    else:
      return printable_text

  # Prints the new text to stdout. If the stream is ending, also prints a newline
  def print_finalized_text(self, text: str, stream_end: bool = False):
    print(text, flush=True, end="" if not stream_end else None)

  ''' Data Pre-/Post-Processing '''
  # Input data padding with initial value
  def init_input_token(self, inputs, max_position):
    padding_num = max_position - inputs.size
    padding_data = -1
    return np.pad(inputs.astype(np.uint32), (0, padding_num), "constant", constant_values=padding_data)

  # Output data with initial value
  def init_output_token(self, max_position):
    return np.full((max_position), -1, np.uint32)

  ''' HRT Buffer Read '''
  # Read single token
  def read_single_token(self, buffer, skip):
    # Calculate address and transfer size of buffer
    size = np.dtype(np.uint32).itemsize
    skip = skip + (size * self.position)
    # Read data from HRT buffer
    data = buffer.read(size=size, skip=skip, dtype=np.uint32)
    return data.item()

class AutoModelForCausalLM:
  
  # Constructor
  def __init__(self, xclbin, ckpt_filedir, inst_filedir, param_filedir, i_filedir, o_filedir, num_device=1) :
    # Set up the path
    self.xclbin = xclbin
    self.ckpt_filedir = ckpt_filedir
    self.inst_filedir = inst_filedir
    self.param_filedir = param_filedir
    self.i_filedir = i_filedir
    self.o_filedir = o_filedir
    # Number of device
    self.num_device = num_device
    # Set GPU Available
    self.slib = "libhdexrt_cu.so" if torch.cuda.is_available() else "libhdexrt_cpu.so"
    # Ring : True, Line : False
    self.search_devices()
    # Empty lists for hrt
    self.device = []
    self.kernel = []
    self.run = []
    self.inst = []
    self.io = []
    self.param = []

  # Search available device
  def search_devices(self) :
    # Find network table
    filename = "/opt/hyperdex/hrt/xclbin/table.json"
    if os.path.exists(filename):
      with open(filename, "r") as f:
        table = json.load(f)
    # If the table does not exist
    else:
      print("[Error\t] Network table does not exist in \"{}\"".format(filename))
      exit(-1)

    # Get the information from the table
    self.num_total_device = table["num_device"]
    self.ring = True if self.num_total_device == self.num_device else False
    self.candidates_fpga_table = [table["network"][i]["src_id"] for i in range(self.num_total_device)]
    self.connectivity_table = [table["network"][i]["direction"] for i in range(self.num_total_device)]
    # Loop the candidates table
    self.candidates_fpga_table.extend(self.candidates_fpga_table[:-1])

    # Check device is in use
    self.available_fpga_table = []
    self.network_topology = []
    for idx, pdidx in enumerate(self.candidates_fpga_table):
      available = hrt.device.usage_check(pdidx=pdidx, slib=self.slib)
      # If the available fpga is found
      if available:
        self.available_fpga_table.append(pdidx)
        # For ring topology
        if self.ring:
          topology = 0 if self.connectivity_table[idx] == "Right" else 1
        # For line topology
        else:
          topology = 2 if self.connectivity_table[idx] == "Right" else 3
        self.network_topology.append(topology)
        # Escape if we collect all fpga
        if len(self.available_fpga_table) == self.num_device:
          break
      # Rest the table if we didn't collect all fpga
      else:
        self.available_fpga_table = []
        self.network_topology = []

    # Check the number of collected fpgas
    if len(self.available_fpga_table) != self.num_device:
      print("[Error\t] Not enough available FPGA! Please wait")
      exit(-1)

  # Setup the device
  def set_devices(self):
    for ldidx, pdidx in enumerate(self.available_fpga_table):
      # Allocate device
      device = hrt.device(pdidx, ldidx, self.slib)
      device.load_xclbin(self.xclbin)
      # Create kernel
      kernel = hrt.kernel(device, self.slib)
      # Append to the lists
      self.device.append(device)
      self.kernel.append(kernel)
      
  # Load model instruction and parameters
  def load_model(self):
    # Read files
    self.read_config_file()
    inst_data = self.read_inst_file()
    actfn_data = self.read_actfn_file()
    param_data = self.read_param_file()
    # Create buffer and load data
    for i in range(self.num_device):
      # Instruction buffer
      inst_size = inst_data.nbytes + actfn_data.nbytes
      inst = hrt.bo(self.device[i], inst_size, self.kernel[i].group_id(hrt.memory_group_flags.inst), self.slib)
      inst.write(inst_data, size=inst_data.nbytes, seek=0, dtype=inst_data.dtype)
      inst.write(actfn_data, size=actfn_data.nbytes, seek=inst_data.nbytes, dtype=actfn_data.dtype)
      # I/O buffer
      io = hrt.bo(self.device[i], 4 * 1024 * 1024, self.kernel[i].group_id(hrt.memory_group_flags.io), self.slib)
      # Parameter buffer
      param_list = []
      for ch in range(32):
        param_size = param_data[i, ch].nbytes
        param = hrt.bo(self.device[i], param_size, self.kernel[i].group_id(hrt.memory_group_flags.param, ch), self.slib)
        param.write(param_data[i, ch], size=param_size, seek=0, dtype=param_data.dtype)
        param_list.append(param)
      # Append to the lists
      self.inst.append(inst)
      self.io.append(io)
      self.param.append(param_list)
    
  # Reload model instruction
  def reload_model_inst(self):
    # Read files
    inst_data = self.read_inst_file()
    actfn_data = self.read_actfn_file()
    # Reload data
    for i in range(self.num_device):
      inst_size = inst_data.nbytes + actfn_data.nbytes
      self.inst[i].write(inst_data, size=inst_data.nbytes, seek=0, dtype=inst_data.dtype)
      self.inst[i].write(actfn_data, size=actfn_data.nbytes, seek=inst_data.nbytes, dtype=actfn_data.dtype)
      # Fetch instruction
      self.run[i].start_reload(self.num_device)
      self.run[i].wait_program()

  # Program device
  def program_device(self):
    for i in range(self.num_device):
      run = hrt.run(self.kernel[i], self.slib)
      run.start_program(self.num_device)
      run.wait_program()
      # Append to the lists
      self.run.append(run)
    
  # Text generation
  def generate(
      self,
      # Generate config
      inputs,
      max_new_tokens=20,
      # Sampling
      do_sample=False,
      top_p=0.9,
      top_k=1,
      temperature=1.0,
      repetition_penalty=1.2,
      seed=None,
      # Streaming
      streamer: TextStreamer=None
    ):
    
    # Generate configuratin
    input_tokens = len(inputs)
    seed = random.randint(2**15-1, 2**31-1) if seed is None else seed
    top_k = 1 if do_sample is False else top_k
    stop = self.eos_token_id
    
    # Batch geneartion (by return keyword)
    if streamer is None:
      # Write input data
      inputs = inputs.astype(np.uint32)
      for i in range(self.num_device):
        self.io[i].write(inputs, size=inputs.nbytes, seek=0, dtype=inputs.dtype)
      # Write frequency table data
      freq_table = np.zeros((self.vocab_size), dtype=np.uint16)
      freq_table[inputs] = 1
      seek = (self.max_position * np.dtype(np.uint32).itemsize) * 2
      for i in range(self.num_device):
        self.io[i].write(freq_table, size=freq_table.nbytes, seek=seek, dtype=freq_table.dtype)
      # Generation call
      for i in range(self.num_device):
        self.run[i].start_generate(self.layer, input_tokens, max_new_tokens, top_p, top_k, temperature, repetition_penalty, stop, seed, self.network_topology[i])
      for i in range(self.num_device):
        self.run[i].wait_generate()
      # Read output data
      skip = (self.max_position * np.dtype(np.uint32).itemsize) + (input_tokens * np.dtype(np.uint32).itemsize)
      generated_token = self.io[0].read(size=max_new_tokens*np.dtype(np.uint32).itemsize, skip=skip, dtype=np.uint32)
      return generated_token

    # Streaming generation (with print keyword)
    else:
      # Write frequency table data
      freq_table = np.zeros((self.vocab_size), dtype=np.uint16)
      freq_table[inputs] = 1
      seek = (self.max_position * np.dtype(np.uint32).itemsize) * 2
      for i in range(self.num_device):
        self.io[i].write(freq_table, size=freq_table.nbytes, seek=seek, dtype=freq_table.dtype)
      # Write input data
      inputs = streamer.init_input_token(inputs, self.max_position)
      for i in range(self.num_device):
        self.io[i].write(inputs, size=inputs.nbytes, seek=0, dtype=inputs.dtype)
      # Initialize output data
      outputs = streamer.init_output_token(self.max_position)
      seek = self.max_position * np.dtype(np.uint32).itemsize
      for i in range(self.num_device):
        self.io[i].write(outputs, size=outputs.nbytes, seek=seek, dtype=outputs.dtype)
      # Generation call
      for i in range(self.num_device):
        self.run[i].start_generate(self.layer, input_tokens, max_new_tokens, top_p, top_k, temperature, repetition_penalty, stop, seed, self.network_topology[i])
      # Read output data
      skip = (self.max_position * np.dtype(np.uint32).itemsize) + (input_tokens * np.dtype(np.uint32).itemsize)
      for i in range(max_new_tokens):
        token = streamer.get(self.io[0], skip)
        streamer.put(token)
        # If genearte ends with stop token
        if token == stop:
          break
      streamer.end()
      # Wait device
      for i in range(self.num_device):
        self.run[i].wait_generate()
      # Read output data
      skip = (self.max_position * np.dtype(np.uint32).itemsize) + (input_tokens * np.dtype(np.uint32).itemsize)
      generated_token = self.io[0].read(size=max_new_tokens*np.dtype(np.uint32).itemsize, skip=skip, dtype=np.uint32)
      return generated_token

  # Text generation with developer mode
  def generate_dev(
    self,
    # Generate config
    num_layer=1,
    i_token_len=10,
    o_token_len=10,
    # Sampling
    do_sample=False,
    top_p=0.9,
    top_k=1,
    temperature=1.0,
    repetition_penalty=1.2,
    seed=None,
  ):
    
    # Generate configuratin
    seed = random.randint(2**15-1, 2**31-1) if seed is None else seed
    top_k = 1 if do_sample is False else top_k
    stop = -1

    # Get the input token data
    for i in range(self.num_device):
      with open("{}/io_{:d}fpga_{:02x}_00.bin".format(self.i_filedir, self.num_device, i), "rb") as f:
        f.seek(np.dtype(np.int32).itemsize * 4, 0)
        data = f.read(self.max_position * np.dtype(np.uint32).itemsize * 2 + self.vocab_size * np.dtype(np.uint16).itemsize)
      inputs = np.frombuffer(data, dtype=np.uint32)
      self.io[i].write(inputs, size=inputs.nbytes, seek=0, dtype=inputs.dtype)
    # Generation call
    for i in range(self.num_device):
      self.run[i].start_generate(num_layer, i_token_len, o_token_len, top_p, top_k, temperature, repetition_penalty, stop, seed, self.network_topology[i])
    for i in range(self.num_device):
      self.run[i].wait_generate()
    # Read output data
    skip = (self.max_position * np.dtype(np.uint32).itemsize) + (i_token_len * np.dtype(np.uint32).itemsize)
    generated_token = self.io[0].read(size=o_token_len*np.dtype(np.uint16).itemsize, skip=skip, dtype=np.uint32)
    # Store output data
    read_size = 4 * 1024 * 1024 # 4MB
    for i in range(self.num_device):
      data = self.io[i].read(size=read_size, skip=0, dtype=np.uint16)
      with open("{}/io_{:d}fpga_{:02x}_00.host.dat".format(self.o_filedir, self.num_device, i), "w") as f:
        for i in range(0, len(data), 32):
          for j in reversed(range(32)):
            f.write("{:04x}".format(data[i+j]))
          f.write("\n")
    return generated_token

  ''' File I/O '''
  # Read configrution file
  def read_config_file(self):
    # Get the model config filename
    filename = os.path.join(self.ckpt_filedir, "hyperdex_config.json")
    with open(filename, "r") as f:
      json_data = json.load(f)
    # Get layer information
    self.layer = json_data["num_hidden_layers"]
    # Get the end of the sequence token and max length
    self.eos_token_id = json_data["eos_token_id"]
    self.max_position = json_data["max_length"]
    self.vocab_size = json_data["vocab_size"]
    # TODO: Get the dim-vector size from the config file
    dim_vector = 64
    if self.vocab_size % dim_vector != 0:
      self.vocab_size = self.vocab_size + (dim_vector - self.vocab_size % dim_vector)
  
  # Read instruction file
  def read_inst_file(self):
    # Get the filename
    filename = os.path.join(self.inst_filedir, "inst_{:d}fpga.bin".format(self.num_device))
    # Read binary file (header size = 16 bytes)
    inst_data = np.fromfile(filename, dtype=np.int32, offset=16)
    return inst_data
  
  # Read activation file
  def read_actfn_file(self):  
    # Get the filename
    filename = os.path.join(self.inst_filedir, "actfn_{:d}fpga.bin".format(self.num_device))
    # Read binary file (header size = 16 bytes)
    actfn_data = np.fromfile(filename, dtype=np.int32, offset=16)
    return actfn_data
      
  # Read parameter file
  def read_param_file(self):
    # Get the data size
    filename = os.path.join(self.param_filedir, "param_{:d}fpga_00_00.bin".format(self.num_device))
    param_size = np.fromfile(filename, dtype=np.uint16, offset=16).size
    # Declare empty paramter
    param_data = np.empty((self.num_device, 32, param_size), dtype=np.uint16)
    for idx in range(param_data.shape[0]):
      for ch in range(param_data.shape[1]):
        # Get the filename
        filename = os.path.join(self.param_filedir, "param_{:d}fpga_{:02x}_{:02x}.bin".format(self.num_device, idx, ch))
        # Read binary file (header size = 16 bytes)
        param_data[idx, ch, :] = np.fromfile(filename, dtype=np.uint16, offset=16)
    return param_data

  # Write parameter file
  def write_param_file(self):
    # Get the data size
    filename = os.path.join(self.param_filedir, "param_{:d}fpga_00_00.bin".format(self.num_device))
    param_size = np.fromfile(filename, dtype=np.uint16, offset=16).nbytes
    # Write parameter data
    for idx in range(self.num_device):
      for ch in range(32):
        # Get the filename
        filename = os.path.join(self.o_filedir, "param_{:d}fpga_{:02x}_{:02x}.bin".format(self.num_device, idx, ch))
        # Write binary file (header size = 16 bytes)
        data = self.param[idx][ch].read(size=param_size, skip=0, dtype=np.uint16)
        with open(filename, "wb") as f:
          f.write(bytes(16))
          f.write(data.tobytes())

  # Model Loader
  @classmethod
  def from_pretrained(cls, xclbin, ckpt_filedir, inst_filedir, param_filedir, i_filedir, o_filedir, id_device=0, num_device=1):
    # Make instance
    model = AutoModelForCausalLM(xclbin, ckpt_filedir, inst_filedir, param_filedir, i_filedir, o_filedir, num_device)
    # Setup device and load model
    model.set_devices()
    model.load_model()
    model.program_device()
    return model

##########################################################################
