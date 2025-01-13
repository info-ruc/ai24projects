import json
import random
from pathlib import Path
import requests
import time

# Groq API 配置
api_url = "https://api.groq.com/openai/v1/chat/completions"
api_key = "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"  # 请确保你的API密钥正确
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

def get_respone_from_api(idx,message):
    payload = {
        "model": "llama3-8b-8192",
        "messages": message,
        "max_tokens": 200,
        "temperature": 0.7
    }
    while True:  # 无限循环，直到任务成功或不再触发速率限制
        start_time = time.time()
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                model_response = result["choices"][0]["message"]["content"].strip()
                print(f"任务{idx + 1}完成，耗时: {end_time - start_time:.2f}s")
                return model_response  # 任务成功，退出当前循环
            elif response.status_code == 429:  # 触发速率限制
                wait_time = float(response.json().get("error", {}).get("message", "").split("in ")[-1].split("ms")[0]) / 1000
                print(f"任务 {idx + 1} 速率限制，等待 {wait_time:.2f}s 重试...")
                time.sleep(wait_time)  # 按照提示的时间等待
            else:
                print(f"任务 {idx + 1} 出错，状态码: {response.status_code}, 响应: {response.text}")
                break  # 其他错误不重试
        except Exception as e:
            print(f"任务 {idx + 1} 出错，网络错误: {str(e)}")
            break
    return 0
