import json
from urllib import request

# This is the ComfyUI api prompt format.
# If you want it for a specific workflow you can "enable dev mode options"
# in the settings of the UI (gear beside the "Queue Size: ") this will enable
# a button on the UI to save workflows in api format.

# Keep in mind ComfyUI is pre alpha software so this format will change a bit.
# This is the one for the default workflow
prompt_text_path = "/home/mabojing/ComfyUI/api_workflows/voice_api.json"

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request("http://10.77.110.170:8188/prompt", data=data)
    request.urlopen(req)

def main():
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the ComfyUI prompt with dynamic inputs.")
    parser.add_argument("ai_say", type=str, help="Text input 1")
    parser.add_argument("text_prompt", type=str, help="Text input 2")
    parser.add_argument("audio_input", type=str, help="Audio input (link to audio file)")
    parser.add_argument("sft_dropdown", type=str, choices=["中文男", "中文女"], help="SFT dropdown value (Chinese male or female)")

    # Parse the arguments
    args = parser.parse_args()

    # Load the default prompt from the JSON file
    prompt = json.load(open(prompt_text_path))

    # Update the prompt with user-provided inputs
    prompt["2"]["inputs"]["text"] = args.ai_say # 输入1
    prompt["12"]["inputs"]["text"] = args.text_prompt # 输入 2
    prompt["13"]["inputs"]["audio"] = args.audio_input # 输入 3
    prompt["15"]["inputs"]["sft_dropdown"] = args.sft_dropdown # 输入 4

    # Send the updated prompt to the ComfyUI queue
    queue_prompt(prompt)

if __name__ == "__main__":
    main()

