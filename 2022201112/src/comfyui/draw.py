import json
from urllib import request, parse
import random

#This is the ComfyUI api prompt format.

#If you want it for a specific workflow you can "enable dev mode options"
#in the settings of the UI (gear beside the "Queue Size: ") this will enable
#a button on the UI to save workflows in api format.

#keep in mind ComfyUI is pre alpha software so this format will change a bit.

#this is the one for the default workflow
prompt_text_path = "/home/mabojing/ComfyUI/api_workflows/flux_draw.json"


def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://10.77.110.170:8188/prompt", data=data)
    request.urlopen(req)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the ComfyUI prompt with dynamic inputs.")
    parser.add_argument("draw_prompt", type=str, help="the prompt for drawing")
    
    args = parser.parse_args()
    
    prompt = json.load(open(prompt_text_path))
    # 先找到id为6的节点
    prompt["6"]["inputs"]["text"] = args.draw_prompt
        
    queue_prompt(prompt)

if __name__ == "__main__":
    main()

