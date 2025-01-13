from diffusers import DiffusionPipeline, LCMScheduler
import torch
import time

SDXL_Models = "Linaqruf/animagine-xl"
LoRA_Models = "lora/arcaea-xl-1.1/checkpoint-15000/pytorch_lora_weights.safetensors"
LCM_LoRA_Models = "latent-consistency/lcm-lora-sdxl"
use_LCM_LoRA = False

pipe = DiffusionPipeline.from_pretrained(
    SDXL_Models,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.load_lora_weights(LoRA_Models, adapter_name="arcaea")

if use_LCM_LoRA:
    pipe.load_lora_weights(LCM_LoRA_Models, adapter_name="lcm")
    pipe.set_adapters(["lcm", "arcaea"], adapter_weights=[0.6, 1.0])
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda", dtype=torch.float16)

prompt = "face focus, cute, masterpiece, best quality, arcaea, hikari, solo, long hair, dress, long hair, dress, white dress, flower, bow, white hair, looking at viewer, hat, long sleeves, very long hair, hair bow, rose, frilled dress, ribbon, outdoors, day, glass fragment"
Guidance_Scale = 0.9

steps = (5, 6, 7, 8, 9, 10, 12) if use_LCM_LoRA else (20, 50)
seed = 1733329668

imgs = []
for step in steps:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    start_time = time.time()
    images = pipe(
        prompt=prompt,
        negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, crow, veil, umbrella, wings, white buret, white bowknots, too many bowknots, red eyes",
        num_inference_steps=step,
        guidance_scale=Guidance_Scale,
        generator=generator
    ).images[0]
    end_time = time.time()
    imgs.append([images, step])

total_time = end_time - start_time
print(f"Total time taken for Generating Image: {total_time} seconds\n")

print(f"seed: {seed}")
for [img, step] in imgs:
    if use_LCM_LoRA:
        img.save(f"img/lcm-lora-{step}step.png")
    else:
        img.save(f"img/lora-{step}step.png")