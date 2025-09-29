import os
import random
import signal
import subprocess
import time

import torch
from openai import OpenAI
from retry import retry

PORT = random.randint(8888, 9999)


class OnlineVLLM:
    """
    Example:
    ```
        vllm = OnlineVLLM(model_id="qwen3.5-72b-instruct", port=8888, devices="0,1,2,3", api_key="empty")
        vllm.init_vllm()
        response = vllm.chat(prompt_messages=[{"role": "user", "content": "Hello, how are you?"}])
        print(response)
        vllm.kill_vllm()
    ```
    """

    def __init__(self, model_id: str, port: int = PORT, devices=None, api_key: str = "empty"):
        """
        Initialize the OnlineVLLM client.

        Args:
            model_id (str): The model identifier to use for inference
            port (int, optional): Port number for the VLLM server. Defaults to 8888.
            devices (str, optional): Comma-separated GPU device IDs. If None, uses all available GPUs.
            api_key (str, optional): API key for OpenAI client. Defaults to "empty".
        """
        self.model_id = model_id
        self.port = port
        self.proc = None
        if devices is None:
            num_of_devices = torch.cuda.device_count()
            self.devices = ",".join([str(i) for i in range(num_of_devices)])
        else:
            self.devices = devices
        self.client = OpenAI(api_key=api_key, base_url=f"http://localhost:{self.port}/v1")

    def is_vllm_ready(self):
        """
        Check if the VLLM server is ready and serving the correct model.

        Returns:
            bool: True if VLLM is ready and serving the expected model, False otherwise
        """
        try:
            models = self.client.models.list()
            if self.model_id != models.data[0].id:
                raise ValueError(f"VLLM for {self.model_id} is not ready")
            return True
        except Exception:
            return False

    def init_vllm(self, **kwargs):
        """
        Initialize and start the VLLM server with the specified model.

        This method starts a VLLM server process if it's not already running.
        It configures GPU memory utilization, model length, and other parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to the VLLM serve command
        """
        if self.is_vllm_ready():
            print(f"VLLM for {self.model_id} is already running on port {self.port}")
            return

        print(f"Running VLLM on {self.devices} for {self.model_id} on port {self.port}")
        max_model_len = 20000
        max_num_seqs = 256
        if any(arg in self.model_id.lower() for arg in ["235"]):
            max_num_seqs = 2
        elif any(arg in self.model_id.lower() for arg in ["120"]):
            max_num_seqs = 4
        elif any(arg in self.model_id.lower() for arg in ["70", "72"]):
            max_num_seqs = 128
        elif any(arg in self.model_id.lower() for arg in ["32", "30", "20"]):
            max_num_seqs = 128
        cmd = f"""
        export CUDA_VISIBLE_DEVICES={self.devices} && \
            vllm serve {self.model_id} \
                -dp {len(self.devices.split(","))} \
                --load-format safetensors \
                --gpu-memory-utilization 0.95 \
                --max-model-len {max_model_len} \
                --max-num-seqs {max_num_seqs} \
                --port {self.port} \
                --enable-prefix-caching \
                --block-size 16"""
        if "qwen3" in self.model_id.lower() and "thinking" in self.model_id.lower():
            cmd += " --reasoning-parser qwen3"

        for key, value in kwargs.items():
            cmd += f" --{key} {value}"

        print("--------------------------------")
        print(cmd)
        print("--------------------------------")
        self.proc = subprocess.Popen(
            cmd,
            shell=True,
            preexec_fn=os.setsid,
        )

        self.wait_for_vllm()

    def wait_for_vllm(self):
        """
        Wait for the VLLM server to become ready.

        This method blocks until the VLLM server is ready to accept requests,
        checking every 10 seconds.
        """
        while not self.is_vllm_ready():
            print(f"VLLM for {self.model_id} is not ready, waiting for 10 seconds...")
            time.sleep(10)

    def kill_vllm(self):
        """
        Terminate the VLLM server process.

        This method stops the VLLM server and waits for it to be fully terminated.
        It sends a SIGTERM signal to the process group and waits for cleanup.
        """
        while self.is_vllm_ready():
            if self.proc is not None:
                os.killpg(self.proc.pid, signal.SIGTERM)

        # if any vllm was running, wait for it to be killed
        if self.proc is not None:
            print("Waiting for VLLM to be killed...")
            time.sleep(30)

    @retry(tries=-1, delay=2, backoff=2)
    def chat(self, prompt_messages: str | list[dict[str, str]], tools=None, tool_choice=None, **kwargs):
        """
        Send a chat completion request to the VLLM server.

        This method sends a request to the VLLM server and returns the response.
        It supports both string prompts and structured message formats.

        Args:
            prompt_messages (str | list[dict[str, str]]): The prompt as a string or list of message dictionaries
            tools (list, optional): List of tools available for the model to use
            tool_choice (str, optional): Tool choice strategy for the model
            **kwargs: Additional parameters to pass to the chat completion API

        Returns:
            ChatCompletion: The response from the VLLM server
        """
        if isinstance(prompt_messages, str):
            prompt_messages = [{"role": "user", "content": prompt_messages}]
        configs = {
            "model": self.model_id,
            "messages": prompt_messages,
        }
        if tools is not None:
            configs["tools"] = tools
        if tool_choice is not None:
            configs["tool_choice"] = tool_choice
        for key, value in kwargs.items():
            configs[key] = value

        configs["timeout"] = 300
        try:
            response = self.client.chat.completions.create(**configs)
        # catch token limit error
        except Exception as e:
            if "token limit" in str(e):
                print("$$$$ erorr during chat: ", e)
                return "regenerate"
            raise e
        # print token
        print(response.usage)
        return response.choices[0].message.content
