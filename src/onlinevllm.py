import os
import signal
import subprocess
import time

import torch
from openai import OpenAI


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

    def __init__(self, model_id: str, port: int = 8888, devices=None, api_key: str = "empty"):
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
        try:
            models = self.client.models.list()
            if self.model_id != models.data[0].id:
                raise ValueError(f"VLLM for {self.model_id} is not ready")
            return True
        except:
            return False

    def init_vllm(self, **kwargs):
        if self.is_vllm_ready():
            print(f"VLLM for {self.model_id} is already running on port {self.port}")
            return

        print(f"Running VLLM on {self.devices} for {self.model_id} on port {self.port}")
        cmd = f"""
        export CUDA_VISIBLE_DEVICES={self.devices} && \
            vllm serve {self.model_id} \
                -tp {len(self.devices.split(","))} \
                --load-format safetensors \
                --gpu-memory-utilization 0.90 \
                --max-model-len 32768 \
                --port {self.port} \
        """
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
        while not self.is_vllm_ready():
            print(f"VLLM for {self.model_id} is not ready, waiting for 10 seconds...")
            time.sleep(10)

    def kill_vllm(self):
        while self.is_vllm_ready():
            if self.proc is not None:
                os.killpg(self.proc.pid, signal.SIGTERM)

        # if any vllm was running, wait for it to be killed
        if self.proc is not None:
            print("Waiting for VLLM to be killed...")
            time.sleep(30)

    def chat(self, prompt_messages: str | list[dict[str, str]], tools=None, tool_choice=None, **kwargs):
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
        response = self.client.chat.completions.create(**configs)
        return response
