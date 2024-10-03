"""Example Python client for `vllm.entrypoints.api_server`
NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend `vllm serve` and the OpenAI client API.
"""

import argparse
import json
from typing import Iterable, List

import requests

import subprocess

import time

# import test_cases



def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(test_case,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    # print(prompt)
    headers = {"User-Agent": "Test Client"}
    # pload = {
    #     "prompt": test_case["prompt"],
    #     "n": n,
    #     "use_beam_search": test_case["use_beam_search"],
    #     "temperature": test_case["temperature"],
    #     "max_tokens": test_case["output_len"], #40,
    #     "top_p": test_case["top_p"],#0.95,
    #     "top_k": test_case["top_k"],#1,
    #     "stream": stream,
    # }
    pload = {
            "prompt": test_case["prompt"],
            "n": test_case["n"],
            "use_beam_search": False,
            "temperature": test_case["temperature"],
            "max_tokens": test_case["output_len"],
            "top_p": test_case["top_p"],
            "top_k": test_case["top_k"],
            "stream": test_case["stream"],
        }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


def main(test_case):
    server_process = subprocess.Popen(f'python -m vllm.entrypoints.api_server --model facebook/{test_case["model_name"]} --device fpga --num-gpu-devices 0 --num-lpu-devices {test_case["num_devices"]}', \
                            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
    # Waiting for server to open
    time.sleep(10)  


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--host", type=str, default="localhost")
    # parser.add_argument("--port", type=int, default=8000)
    # parser.add_argument("--n", type=int, default=1)

    # # parser.add_argument("--prompt", type=str, default="Hello, my name is")
    # parser.add_argument("--prompt", type=str, default=test_case["prompt"])

    # parser.add_argument("--stream", action="store_true")
    # args = parser.parse_args()
    # prompt = args.prompt
    host = "localhost"
    port = 8000
    api_url = "http://localhost:8000/generate"
    # api_url = f"http://{args.host}:{args.port}/generate"
    # n = args.n
    # stream = args.stream

    # print(f"Prompt: {prompt!r}\n", flush=True)
    stream_in = False
    response = post_http_request(test_case, api_url, n=1, stream=stream_in)

    if stream_in:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                # print(f"Beam candidate {i}: {line!r}", flush=True)
                return line
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            # print(f"Beam candidate {i}: {line!r}", flush=True)
            return line

    subprocess.Popen.kill(server_process)
    time.sleep(10) 




if __name__ == "__main__":

    server_process = subprocess.Popen(f'python -m vllm.entrypoints.api_server --model facebook/opt-1.3b --device fpga --num-gpu-devices 0 --num-lpu-devices 2', \
                            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
    # Waiting for server to open
    time.sleep(10)  


    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)

    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    # parser.add_argument("--prompt", type=str, default=test_case[""])

    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n
    stream = args.stream

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url, n, stream)

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)

    subprocess.Popen.kill(server_process)
    time.sleep(10) 

############################################################################################33


# import subprocess
# import requests

# import time


# """Example Python client for `vllm.entrypoints.api_server`
# NOTE: The API server is used only for demonstration and simple performance
# benchmarks. It is not intended for production use.
# For production use, we recommend `vllm serve` and the OpenAI client API.
# """

# import argparse
# import json
# from typing import Iterable, List

# import requests


# class TextGenerator:

#     def __init__(self, test_case):
#         self.test_case = test_case

#     def start(self):

#         server_process = subprocess.Popen(f'python -m vllm.entrypoints.api_server --model {self.test_case["model_name"]} --device fpga --num-gpu-devices 0 --num-lpu-devices {self.test_case["num_devices"]}', \
#                                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
    
#         # Waiting for server to open
#         time.sleep(10)


#         response = self.post_http_request()
#         if self.test_case["stream"]:
#             num_printed_lines = 0
#             for h in self.get_streaming_response(response):
#                 self.clear_line(num_printed_lines)
#                 num_printed_lines = 0
#                 for i, line in enumerate(h):
#                     num_printed_lines += 1
#                     #print(f"Beam candidate {i}: {line!r}", flush=True)
#                     print(line)
#                     return line
#         else:
#             output = self.get_response(response)
#             for i, line in enumerate(output):
#                 #print(f"Beam candidate {i}: {line!r}", flush=True)
#                 print(line)
#                 return line
#         #server_process.wait()   
#         #time.sleep(10)  
#         subprocess.Popen.kill(server_process)
#         # server_process.terminate()
          





#     def clear_line(self, n: int = 1) -> None:
#         LINE_UP = '\033[1A'
#         LINE_CLEAR = '\x1b[2K'
#         for _ in range(n):
#             print(LINE_UP, end=LINE_CLEAR, flush=True)

#     def post_http_request(self):
#         api_url = "http://" + f"{self.test_case['host']}:{self.test_case['port']}/generate"
        
#         # print(prompt)
#         headers = {"User-Agent": "Test Client"}
#         pload = {
#             "prompt": self.test_case["prompt"],
#             "n": self.test_case["n"],
#             "use_beam_search": False,
#             "temperature": self.test_case["temperature"],
#             "max_tokens": self.test_case["output_len"],
#             "top_p": self.test_case["top_p"],
#             "top_k": self.test_case["top_k"],
#             "stream": self.test_case["stream"],
#         }
#         response = requests.post(api_url,
#                                 headers=headers,
#                                 json=pload,
#                                 stream=self.test_case["stream"])
#         return response

#     def get_streaming_response(self, response: requests.Response) -> Iterable[List[str]]:
#         for chunk in response.iter_lines(chunk_size=8192,
#                                         decode_unicode=False,
#                                         delimiter=b"\0"):
#             if chunk:
#                 data = json.loads(chunk.decode("utf-8"))
#                 output = data["text"]
#                 yield output


#     def get_response(self, response: requests.Response) -> List[str]:
#         data = json.loads(response.content)
#         output = data["text"]
#         return output



# def main(test_case):
#   generator = TextGenerator(test_case)
#   answer = generator.start()
#   #time.sleep(10000000000)
#   return answer



# #################################################
# # def clear_line(n: int = 1) -> None:
# #     LINE_UP = '\033[1A'
# #     LINE_CLEAR = '\x1b[2K'
# #     for _ in range(n):
# #         print(LINE_UP, end=LINE_CLEAR, flush=True)


# # def post_http_request(prompt: str,
# #                       api_url: str,
# #                       n: int = 1,
# #                       stream: bool = False) -> requests.Response:
# #     print(prompt)
# #     headers = {"User-Agent": "Test Client"}
# #     pload = {
# #         "prompt": prompt,
# #         "n": n,
# #         "use_beam_search": False,
# #         "temperature": 0.8,
# #         "max_tokens": 40,
# #         "top_p": 0.95,
# #         "top_k": 1,
# #         "stream": stream,
# #     }
# #     response = requests.post(api_url,
# #                              headers=headers,
# #                              json=pload,
# #                              stream=stream)
# #     return response


# # def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
# #     for chunk in response.iter_lines(chunk_size=8192,
# #                                      decode_unicode=False,
# #                                      delimiter=b"\0"):
# #         if chunk:
# #             data = json.loads(chunk.decode("utf-8"))
# #             output = data["text"]
# #             yield output


# # def get_response(response: requests.Response) -> List[str]:
# #     data = json.loads(response.content)
# #     output = data["text"]
# #     return output
# #################################################




# #if __name__ == "__main__":

#     # Opening serving system in background
#     # server_process = subprocess.Popen('python -m vllm.entrypoints.api_server --model facebook/opt-1.3b --device fpga --num-gpu-devices 0 --num-lpu-devices 2', \
#     #                               shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
    
#     # # Waiting for server to open
#     # time.sleep(5)


#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--host", type=str, default="localhost")
#     # parser.add_argument("--port", type=int, default=8000)
#     # parser.add_argument("--n", type=int, default=1)
#     # #parser.add_argument("--prompt", type=str, default="1 2 3 4 5")
#     # parser.add_argument("--prompt", type=str, default="Hello, my name is")
#     # parser.add_argument("--stream", action="store_true")
#     # args = parser.parse_args()
#     # prompt = args.prompt
#     # api_url = f"http://{args.host}:{args.port}/generate"
#     # n = args.n
#     # stream = args.stream

#     # print(f"Prompt: {prompt!r}\n", flush=True)
#     # response = post_http_request(prompt, api_url, n, stream)

#     # if stream:
#     #     num_printed_lines = 0
#     #     for h in get_streaming_response(response):
#     #         clear_line(num_printed_lines)
#     #         num_printed_lines = 0
#     #         for i, line in enumerate(h):
#     #             num_printed_lines += 1
#     #             print(f"Beam candidate {i}: {line!r}", flush=True)
#     # else:
#     #     output = get_response(response)
#     #     for i, line in enumerate(output):
#     #         print(f"Beam candidate {i}: {line!r}", flush=True)


#     #Kill serving system
#     # subprocess.Popen.kill(server_process)
