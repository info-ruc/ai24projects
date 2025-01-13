from diffusers import DiffusionPipeline, LCMScheduler, EulerAncestralDiscreteScheduler
import torch
import time

SDXL_Models = "Linaqruf/animagine-xl"
LoRA_Models_old = "sheriyuo/animagine-xl-arcaea-lora-draw"
LoRA_Models = "sheriyuo/animagine-xl-arcaea-lora"
LCM_LoRA_Models = "latent-consistency/lcm-lora-sdxl"
use_LCM_LoRA = False
mix_LoRA = False

pipe = DiffusionPipeline.from_pretrained(
    SDXL_Models,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.load_lora_weights(LoRA_Models, adapter_name="arcaea")

if mix_LoRA:
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LoRA_Models_old, adapter_name="paint")
    pipe.set_adapters(["arcaea", "paint"], adapter_weights=[0.8, 0.2])

if use_LCM_LoRA:
    pipe.load_lora_weights(LCM_LoRA_Models, adapter_name="lcm")
    pipe.set_adapters(["arcaea", "lcm"], adapter_weights=[1.0, 0.8])
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda", dtype=torch.float16)

prompt = "face focus, cute, masterpiece, best quality, arcaea, hikari, solo, long hair, dress, long hair, dress, white dress, flower, bow, white hair, looking at viewer, hat, long sleeves, very long hair, hair bow, rose, frilled dress, ribbon, outdoors, day, glass fragment"
No_of_Steps = 10 if use_LCM_LoRA else 50
Guidance_Scale = 0.9

num = 1

imgs = []
for _ in range(num):
    seed = 1734704641
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    start_time = time.time()
    images = pipe(
        prompt=prompt,
        negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username",
        num_inference_steps=No_of_Steps,
        guidance_scale=Guidance_Scale,
        generator=generator
    ).images[0]
    end_time = time.time()
    imgs.append([images, seed])

total_time = end_time - start_time
print(f"Total time taken for Generating Image: {total_time} seconds\n")

print(f"seed: {seed}")
if num == 1:
    for [img, seed] in imgs:
        if use_LCM_LoRA:
            img.save(f"img/lcm-lora/{seed}.png")
        else:
            img.save(f"img/lora/{seed}.png")
else:
    images.save("image.png")