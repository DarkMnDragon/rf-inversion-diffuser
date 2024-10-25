import torch
import math
import argparse
import os

from PIL import Image
from diffusers import FluxPipeline
from torch import Tensor
from torchvision import transforms

def decode_imgs(latents, pipeline):
    with torch.inference_mode():
        imgs = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        imgs = pipeline.vae.decode(imgs)[0]
        imgs = pipeline.image_processor.postprocess(imgs, output_type="pil")
    return imgs

def encode_imgs(imgs, pipeline, DTYPE):
    with torch.inference_mode():
        latents = pipeline.vae.encode(imgs).latent_dist.sample()
        latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
        latents = latents.to(dtype=DTYPE)
    return latents

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float32)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def interpolated_inversion(
    pipeline, 
    timesteps, 
    latents,
    gamma,
    DTYPE,
):
    with torch.inference_mode():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt( # null text
            prompt="", 
            prompt_2=""
        )
        latent_image_ids = pipeline._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2],
            latents.shape[3],
            latents.device, 
            DTYPE,
        )
        packed_latents = pipeline._pack_latents(
            latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )
        
        target_noise = torch.randn(packed_latents.shape, device=packed_latents.device, dtype=torch.float32)
        guidance_scale=0.0 # zero guidance for inversion
        guidance_vec = torch.full((packed_latents.shape[0],), guidance_scale, device=packed_latents.device, dtype=packed_latents.dtype)

        # Image inversion with interpolated velocity field.  t goes from 0.0 to 1.0
        with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)
                
                # Null text velocity
                flux_velocity = pipeline.transformer(
                        hidden_states=packed_latents,
                        timestep=t_vec,
                        guidance=guidance_vec,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=pipeline,
                    )[0]
                
                # Prevents precision issues
                packed_latents = packed_latents.to(torch.float32)
                flux_velocity = flux_velocity.to(torch.float32)

                # Target noise velocity
                target_noise_velocity = (target_noise - packed_latents) / (1.0 - t_curr)
                
                # interpolated velocity
                interpolated_velocity = gamma * target_noise_velocity + (1 - gamma) * flux_velocity
                
                # one step Euler
                packed_latents = packed_latents + (t_prev - t_curr) * interpolated_velocity
                
                packed_latents = packed_latents.to(DTYPE)
                progress_bar.update()
                
        print("Mean Absolute Error", torch.mean(torch.abs(packed_latents - target_noise)))
        
        latents = pipeline._unpack_latents(
                packed_latents,
                height=1024,
                width=1024,
                vae_scale_factor=pipeline.vae_scale_factor,
        )
        latents = latents.to(DTYPE)
    return latents

def interpolated_denoise(
    pipeline, 
    timesteps, 
    inversed_latents,
    img_latents,
    eta,
    start_time,
    end_time,
    use_inversed_latents=True,
    guidance_scale=3.5,
    prompt='photo of a tiger',
    DTYPE=torch.bfloat16,
):
    with torch.inference_mode():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, 
            prompt_2=prompt
        )
        latent_image_ids = pipeline._prepare_latent_image_ids(
            inversed_latents.shape[0],
            inversed_latents.shape[2],
            inversed_latents.shape[3],
            inversed_latents.device,
            DTYPE,
        )
        if use_inversed_latents:
            packed_latents = pipeline._pack_latents(
                inversed_latents,
                batch_size=inversed_latents.shape[0],
                num_channels_latents=inversed_latents.shape[1],
                height=inversed_latents.shape[2],
                width=inversed_latents.shape[3],
            )
        else:
            tmp_latents = torch.randn_like(inversed_latents)
            packed_latents = pipeline._pack_latents(
                tmp_latents,
                batch_size=tmp_latents.shape[0],
                num_channels_latents=tmp_latents.shape[1],
                height=tmp_latents.shape[2],
                width=tmp_latents.shape[3],
            )
    
        packed_img_latents = pipeline._pack_latents(
            img_latents,
            batch_size=img_latents.shape[0],
            num_channels_latents=img_latents.shape[1],
            height=img_latents.shape[2],
            width=img_latents.shape[3],
        )
        
        target_img = packed_img_latents.clone().to(torch.float32)
        guidance_vec = torch.full((packed_latents.shape[0],), guidance_scale, device=packed_latents.device, dtype=packed_latents.dtype)

        # Denoising with interpolated velocity field.  t goes from 1.0 to 0.0
        with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)
                
                # Editing text velocity
                flux_velocity = pipeline.transformer(
                        hidden_states=packed_latents,
                        timestep=t_vec,
                        guidance=guidance_vec,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=pipeline,
                    )[0]
                
                # Prevents precision issues
                packed_latents = packed_latents.to(torch.float32)
                flux_velocity = flux_velocity.to(torch.float32)

                # Target image velocity
                target_img_velocity = -(target_img - packed_latents) / t_curr
                
                # interpolated velocity
                if end_time <= t_curr <= start_time:
                    interpolated_velocity = eta * target_img_velocity + (1 - eta) * flux_velocity
                    packed_latents = packed_latents + (t_prev - t_curr) * interpolated_velocity
                    print(f"X_{t_prev:.3f} = X_{t_curr:.3f} + {t_prev - t_curr:.3f} * interpolated velocity")
                else:
                    packed_latents = packed_latents + (t_prev - t_curr) * flux_velocity
                    print(f"X_{t_prev:.3f} = X_{t_curr:.3f} + {t_prev - t_curr:.3f} * flux velocity")
                
                packed_latents = packed_latents.to(DTYPE)
                progress_bar.update()
        
        latents = pipeline._unpack_latents(
                packed_latents,
                height=1024,
                width=1024,
                vae_scale_factor=pipeline.vae_scale_factor,
        )
        latents = latents.to(DTYPE)
    return latents

def main():
    parser = argparse.ArgumentParser(description='Test interpolated_denoise with different parameters.')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/Flux-dev', help='Path to the pretrained model')
    parser.add_argument('--image_path', type=str, default='/root/rf-inversion/example/cat.png', help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output images')
    parser.add_argument('--eta', type=float, default=1.0, help='Eta parameter for interpolated_denoise')
    parser.add_argument('--start_time', type=float, default=1.0, help='Start time for interpolated_denoise')
    parser.add_argument('--end_time', type=float, default=0.87, help='End time for interpolated_denoise')
    parser.add_argument('--use_inversed_latents', action='store_true', help='Use inversed latents')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='Guidance scale for interpolated_denoise')
    parser.add_argument('--num_steps', type=int, default=28, help='Number of steps for timesteps')
    parser.add_argument('--shift', action='store_true', help='Use shift in get_schedule')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter for interpolated_inversion')
    parser.add_argument('--prompt', type=str, default='photo of a tiger', help='Prompt text for generation')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'], help='Data type for computations')

    args = parser.parse_args()

    if args.dtype == 'bfloat16':
        DTYPE = torch.bfloat16
    elif args.dtype == 'float16':
        DTYPE = torch.float16
    elif args.dtype == 'float32':
        DTYPE = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=DTYPE)
    pipe.to(device)

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess the image
    img = Image.open(args.image_path)

    train_transforms = transforms.Compose(
                [
                    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(1024),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    img = train_transforms(img).unsqueeze(0).to(device).to(DTYPE)

    # Encode image to latents
    img_latent = encode_imgs(img, pipe, DTYPE)

    # Generate timesteps for inversion
    timesteps_inversion = get_schedule( 
                    num_steps=50,
                    image_seq_len=(1024 // 16) * (1024 // 16), # vae_scale_factor = 16
                    shift=False,  # Set True for Flux-dev, False for Flux-schnell
                )[::-1] # flipped for inversion

    inversed_latent = interpolated_inversion(pipe, timesteps_inversion, img_latent, gamma=args.gamma, DTYPE=DTYPE)

    # Generate timesteps for denoising
    timesteps_denoise = get_schedule(
    				num_steps=args.num_steps,
    				image_seq_len=(1024 // 16) * (1024 // 16), # vae_scale_factor = 16
    				shift=args.shift,
    			)

    # Denoise
    img_latents = interpolated_denoise(
        pipe, 
    	timesteps_denoise,
    	inversed_latent,
    	img_latent,
    	eta=args.eta,
    	start_time=args.start_time,
    	end_time=args.end_time,
        use_inversed_latents=args.use_inversed_latents,
        guidance_scale=args.guidance_scale,
        prompt=args.prompt,
        DTYPE=DTYPE,
    )

    # Decode latents to images
    out = decode_imgs(img_latents, pipe)[0]

    # Save output image
    output_filename = f"eta{args.eta}_start{args.start_time}_end{args.end_time}_guidance{args.guidance_scale}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    out.save(output_path)
    print(f"Saved output image to {output_path} with parameters: eta={args.eta}, start_time={args.start_time}, end_time={args.end_time}, guidance_scale={args.guidance_scale}")

if __name__ == "__main__":
    main()
