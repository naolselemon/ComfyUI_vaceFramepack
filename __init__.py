import torch
import numpy as np
from comfy import model_management as mm
import math
import gc
from tqdm import tqdm
import torch.nn.functional as F
from comfy.utils import ProgressBar, common_upscale
from .framepack_helpers import (
    BenchmarkManager,
    PromptHandler,
    SchedulerFactory,
    RoPEEmbeddings,
    VAEProcessor,
    ContextBuilder,
    MaskGenerator,
    FrequencyProcessor,
    ReferenceImageProcessor,
    VAE_STRIDE,
    SparseSelector,
    MoCRouter,
    FramePackCompressor,
    VideoMetrics,
    BenchmarkAnalyzer
)
import time
from diffusers.schedulers import DEISMultistepScheduler
from .wanvideo.utils.basic_flowmatch import FlowMatchScheduler


class WanVACEVideoFramepackSampler2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "vae": ("WANVAE",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["dpm++", "unipc", "euler", "deis", "lcm"], {"default": "unipc"}),
                "num_frames": ("INT", {"default": 121, "min": 41, "max": 1000, "step": 1}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "n_ref_frames": ("INT", {"default": 1, "min": 1, "max": 121, "step": 1}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "context_method": (["contiguous", "sparse", "moc", "frame"], {"default": "contiguous"}),
                "num_context_chunks": ("INT", {"default": 5, "min": 1, "max": 20}),
                "lambda_compression": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "top_k_chunks": ("INT", {"default": 3, "min": 1, "max": 30}),
                "context_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "tiled_vae": ("BOOLEAN", {"default": True}),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
              
            },
            "optional": {
                "sigmas": ("SIGMAS",),
                "ref_images": ("IMAGE",),
                "input_frames": ("VIDEO",),
                "input_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT", "VIDEO")
    RETURN_NAMES = ("samples", "decoded_video")
    FUNCTION = "process"
    CATEGORY = "framepackVACE"
    DESCRIPTION = "A sampler specifically for the FramePack algorithm for long video generation using hierarchical context."

    def __init__(self):
        self.vae_processor = None
        self.frame_compressor = None
        self.device = None
        self.cache_state = None
        self.benchmark_manager = BenchmarkManager()
        # Optimize CUDA performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def process(self, model, vae, steps, cfg, shift, seed, scheduler,
                num_frames, width, height, n_ref_frames, force_offload,
                context_method, num_context_chunks, lambda_compression, top_k_chunks,
                context_strength, tiled_vae=True,
                text_embeds=None,
                ref_images=None, input_frames=None, input_mask=None,
                sigmas=None):

        if text_embeds is None:
            raise ValueError("text_embeds (from WanVideo TextEncoder) is required")

        enable_benchmarking = True
        benchmark_output_dir = "./benchmarks"

        if enable_benchmarking:
            self.benchmark_manager.overall_start_time = time.time()
            self.benchmark_manager.generation_params = {
                'num_frames': num_frames,
                'width': width,
                'height': height,
                'steps': steps,
                'cfg': cfg,
                'scheduler': scheduler,
                'seed': seed,
            }
            print("\n🔬 Benchmarking enabled - tracking performance metrics...")

        device = mm.get_torch_device()
        self.device = device
        offload_device = mm.unet_offload_device()

        model_obj = model.model
        model_wrapper = model_obj.diffusion_model

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.vae_processor = VAEProcessor(vae.to(device).to(dtype), device)
        self.frame_compressor = FramePackCompressor(lambda_compression=lambda_compression)
        model_wrapper.to(device)

        width = (width // 16) * 16
        height = (height // 16) * 16

        INITIAL_FRAMES = 121
        num_sections = 1 if num_frames <= INITIAL_FRAMES else math.ceil(num_frames / INITIAL_FRAMES)


        text_embeds_list = [text_embeds] * num_sections

        print(f"[FramePack] Using the same WanVideo text embeds repeated across {num_sections} sections.")

        section_prompts = ["WanVideo encoded prompt"] * num_sections  # placeholder for benchmark report

        print(f"\n[DEBUG] Using {len(text_embeds_list)} embed sections (repeated single prompt).")

        latents = self._generate_with_framepack_multi(
            model_wrapper=model_wrapper,
            section_text_embeds=text_embeds_list,
            input_frames=input_frames,
            input_masks=input_mask,
            ref_images=ref_images,
            width=width,
            height=height,
            num_frames=num_frames,
            shift=shift,
            scheduler_name=scheduler,
            context_method=context_method,
            num_context_chunks=num_context_chunks,
            lambda_compression=lambda_compression,
            top_k_chunks=top_k_chunks,
            context_strength=context_strength,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sigmas=sigmas,
            device=device,
            offload_device=offload_device,
            force_offload=force_offload,
            tiled_vae=tiled_vae,
            n_ref_frames=n_ref_frames
        )

        if enable_benchmarking:
            report = self.benchmark_manager.generate_report(section_prompts)
            print("\n" + report)
            self.benchmark_manager.save_report(report, benchmark_output_dir)

        # Note: your original code only returned latents — adjust if you want to actually decode & return video
        return ({"samples": latents.unsqueeze(0).cpu()}, )

    def _generate_with_framepack_multi(self, model_wrapper, section_text_embeds,
                                       input_frames, input_masks,
                                       ref_images, width, height, num_frames,
                                       shift, scheduler_name,
                                       context_method, num_context_chunks,
                                       lambda_compression, top_k_chunks, context_strength,
                                       steps, cfg, seed, sigmas,
                                       device, offload_device, force_offload, tiled_vae=True,
                                       n_ref_frames=1):
        """Core FramePack generation algorithm with multi-prompt support"""
        vae_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        all_generated_latents = []
        accumulated_latents = []
        total_output_frames = 0

        LATENT_WINDOW = 60
        GENERATION_FRAMES = 30
        CONTEXT_FRAMES = 30
        INITIAL_FRAMES = 121

        analyzer = BenchmarkAnalyzer()
        reference_character_embed = None
        num_sections = 1 if num_frames <= INITIAL_FRAMES else math.ceil(num_frames / INITIAL_FRAMES)

        for section in range(num_sections):
            print(f"\n[Section {section+1}/{num_sections}]")
            print(f"Using WanVideo pre-encoded embeds for section {section+1}")
            text_embeds = section_text_embeds[section]

       

            self.benchmark_manager.benchmark_section(section, 'encoding')

            if section == 0:
                input_frames = torch.zeros(1, 3, INITIAL_FRAMES, height, width,
                                           device=device, dtype=vae_dtype)
                input_masks = torch.ones_like(input_frames, device=device, dtype=vae_dtype)
                input_frames = [(p * 2 - 1) for p in input_frames]
                print(f"[DEBUG] Section 0: Input frames prepared. Shape: {input_frames[0].shape}")

                if ref_images is not None:
                    ref_images = ReferenceImageProcessor.process_reference_images(
                        ref_images, width, height, device, vae_dtype, n_ref_frames
                    )

                z0 = self.vae_processor.encode_frames(input_frames, ref_images=ref_images,
                                                      masks=input_masks, tiled_vae=tiled_vae)
                m0 = self.vae_processor.encode_masks(input_masks, ref_images=ref_images)
                z = self.vae_processor.combine_latent(z0, m0)

                target_shape = (
                    16,
                    (INITIAL_FRAMES - 1) // VAE_STRIDE[0] + 1,
                    height // VAE_STRIDE[1],
                    width // VAE_STRIDE[2]
                )
            else:
                mm.soft_empty_cache()
                gc.collect()

                if context_method == "frame":
                    print(f"Using Frame (Pseudo FramePack) context management")
                    z_context = self.frame_compressor.prepare_context(accumulated_latents, section)
                elif context_method == "sparse":
                    print(f"Using Sparse (Anchor-Based) context management")
                    z_context = SparseSelector.pick_sparse_context(accumulated_latents, CONTEXT_FRAMES)
                elif context_method == "moc":
                    print(f"Using MoC (Mixture of Contexts) context management")
                    z_context = MoCRouter.retrieve_context(accumulated_latents, section_text_embeds[section], top_k=top_k_chunks)
                else:  # contiguous
                    print(f"Using Contiguous context management")
                    z_context = ContextBuilder.pick_context(torch.cat(accumulated_latents, dim=1), section)

                print(f"Context latent shape: {z_context.shape}")
                blend_val = 0.05
                u = z_context * (1.0 - blend_val)
                c = z_context * blend_val
                m_vace = torch.ones((64, z_context.shape[1], z_context.shape[2], z_context.shape[3]),
                                    device=device, dtype=vae_dtype) * blend_val
                z_bypass = torch.cat([u, c, m_vace], dim=0)
                z = [z_bypass]
                print(f"Bypassing VAE for context. 96-ch Latent shape: {z[0].shape}")
                print(f"[DEBUG] Context Stats: Mean={z[0].mean().item():.6f}, Std={z[0].std().item():.6f}")

            self.benchmark_manager.benchmark_section(section, 'encoding')

          
            # Denoising phase (unchanged)
          
            self.benchmark_manager.benchmark_section(section, 'denoising')

            sample_scheduler = SchedulerFactory.create_scheduler(
                scheduler_name, steps, shift, device, sigmas
            )
            timesteps = sample_scheduler.timesteps

            generator = torch.Generator(device="cpu")
            generator.manual_seed((seed + section) if seed != -1 else torch.randint(0, 2**32, (1,)).item())

            has_ref = ref_images is not None
            noise = torch.randn(
                16,
                z[0].shape[1],
                z[0].shape[2],
                z[0].shape[3],
                dtype=vae_dtype,
                device="cpu",
                generator=generator
            )
            latent = noise.to(device)

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])
            freqs = RoPEEmbeddings.setup_rope_embeddings(model_wrapper, latent.shape[1])
            num_steps = len(timesteps)
            effective_context_strength = 1 if section == 0 else context_strength
            vace_data = [{
                "context": z,
                "scale": [effective_context_strength] * num_steps,
                "start": 0.0,
                "end": 1.0,
                "seq_len": seq_len
            }]

            if not isinstance(cfg, list):
                cfg = [cfg] * (steps + 1)

            pbar = ProgressBar(steps)
            mm.soft_empty_cache()
            gc.collect()

            self.cache_state = [None, None]

            for idx, t in enumerate(timesteps):
                print(idx+1, 'of ', num_steps)
                timestep = torch.tensor([t]).to(device)

                noise_pred = self._predict_with_cfg(
                    latent=latent,
                    cfg_scale=cfg[idx],
                    text_embeds=text_embeds,
                    timestep=timestep,
                    idx=idx,
                    model_wrapper=model_wrapper,
                    vace_data=vace_data,
                    seq_len=seq_len,
                    freqs=freqs,
                    device=device
                )

                step_args = {"generator": generator}
                if isinstance(sample_scheduler, (DEISMultistepScheduler, FlowMatchScheduler)):
                    step_args.pop("generator", None)

                latent = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    **step_args
                )[0].squeeze(0)

                pbar.update(1)

                if force_offload and idx % 10 == 0:
                    mm.soft_empty_cache()

            self.benchmark_manager.benchmark_section(section, 'denoising')

           
            # Accumulation & metrics 
            
            self.benchmark_manager.benchmark_section(section, 'accumulation')

            if section == 0:
                ref_len = latent.shape[1] - target_shape[1]
                if ref_len > 0:
                    latent_without_ref = latent[:, ref_len:, :, :]
                else:
                    latent_without_ref = latent
                accumulated_latents.append(latent_without_ref)
                all_generated_latents.append(latent_without_ref)
            else:
                if section > 2:
                    accumulated_latents.pop(0)
                new_content = latent[:, -GENERATION_FRAMES:, :, :]
                accumulated_latents.append(new_content)
                all_generated_latents.append(new_content)
                frames_added = new_content.shape[1]
                total_output_frames += frames_added
                print(f"Added {frames_added} frames (total: {total_output_frames})")

            # Optional evaluation block (unchanged) ...
            try:
                if section == 0 and reference_character_embed is None:
                    frame_ref = self.vae_processor.decode_single_frame(all_generated_latents[0], index=0)
                    reference_character_embed = all_generated_latents[0][:, 0, :, :].mean(dim=(1, 2))
                    print("Captured reference character embedding for identity tracking.")

                if section > 0:
                    frame_prev = self.vae_processor.decode_single_frame(all_generated_latents[-2], index=-1)
                    frame_curr = self.vae_processor.decode_single_frame(all_generated_latents[-1], index=0)
                    ssim_val = VideoMetrics.calculate_ssim_boundary(frame_prev, frame_curr)
                    self.benchmark_manager.log_metric(section, "boundary_ssim", ssim_val)
                    print(f"Boundary SSIM (Section {section-1} -> {section}): {ssim_val:.4f}")

                    current_embed = all_generated_latents[-1][:, 0, :, :].mean(dim=(1, 2))
                    drift = VideoMetrics.calculate_embedding_drift(reference_character_embed, current_embed)
                    self.benchmark_manager.log_metric(section, "identity_drift", drift)
                    print(f"Identity Drift: {drift:.4f}")
            except Exception as e:
                print(f"Metrics calculation error: {e}")

            self.benchmark_manager.benchmark_section(section, 'accumulation')

            if 'noise_pred' in locals():
                del latent, noise_pred
            mm.soft_empty_cache()
            gc.collect()

        if force_offload:
            model_wrapper.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        analyzer.save_run_data(context_method, self.benchmark_manager)
        report_md = analyzer.generate_comparison_report()
        print("\n" + report_md)

        final_latent = torch.cat(all_generated_latents, dim=1)
        return final_latent.cpu()

    def _predict_with_cfg(self, latent, cfg_scale, text_embeds, timestep, idx,
                          model_wrapper, vace_data, seq_len, freqs, device):
        """Classifier-free guidance prediction"""
        try:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        except:
            dtype = torch.float16

        latent = latent.to(dtype)
        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype):
            base_params = {
                'seq_len': seq_len,
                'device': device,
                'freqs': freqs,
                't': timestep,
                'current_step': idx,
                "nag_params": text_embeds.get("nag_params", {}),
                "nag_context": text_embeds.get("nag_prompt_embeds", None),
                "ref_target_masks": None
            }
            current_step_percentage = idx / 30

            noise_pred_cond, cache_state_cond = model_wrapper(
                [latent],
                context=text_embeds["prompt_embeds"],
                y=None,
                clip_fea=None,
                is_uncond=False,
                current_step_percentage=current_step_percentage,
                pred_id=self.cache_state[0] if self.cache_state else None,
                vace_data=vace_data,
                attn_cond=None,
                **base_params
            )
            noise_pred_cond = noise_pred_cond[0]

            if math.isclose(cfg_scale, 1.0):
                self.cache_state = [cache_state_cond, None]
                return noise_pred_cond

            noise_pred_uncond, cache_state_uncond = model_wrapper(
                [latent],
                context=text_embeds["negative_prompt_embeds"],
                y=None,
                clip_fea=None,
                is_uncond=True,
                current_step_percentage=current_step_percentage,
                pred_id=self.cache_state[1] if self.cache_state else None,
                vace_data=vace_data,
                attn_cond=None,
                **base_params
            )
            noise_pred_uncond = noise_pred_uncond[0]

            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            self.cache_state = [cache_state_cond, cache_state_uncond]
            return noise_pred


NODE_CLASS_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": WanVACEVideoFramepackSampler2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": "WanVACE FramePack Sampler 2"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']