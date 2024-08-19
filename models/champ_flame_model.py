#####################################
# Made by VF
# modifed from : https://github.com/fudan-generative-vision/champ/blob/master/models/champ_model.py
#


import torch
import torch.nn as nn
from models.unet_2d_condition import UNet2DConditionModel


class ChampFlameModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        # denoising_unet: UNet3DConditionModel,
        reference_control_writer,
        # reference_control_reader,
        guidance_encoder_group,
    ):
        
        super().__init__()
        self.reference_unet = reference_unet
        # self.denoising_unet = denoising_unet

        self.reference_control_writer = reference_control_writer
       # self.reference_control_reader = reference_control_reader

        self.guidance_types = []
        self.guidance_input_channels = []

#########################
# VF : you need to look for the guidance input
        for guidance_type, guidance_module in guidance_encoder_group.items():
            setattr(self, f"guidance_encoder_{guidance_type}", guidance_module)
            self.guidance_types.append(guidance_type)
            self.guidance_input_channels.append(guidance_module.guidance_input_channels)
    
    def forward(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        # ref_image_latents,
        multi_guidance_cond,
      #   uncond_fwd: bool = False,
    ):
        guidance_cond_group = torch.split(
            multi_guidance_cond, self.guidance_input_channels, dim=1
        )
        guidance_fea_lst = []

        for guidance_idx, guidance_cond in enumerate(guidance_cond_group):
            guidance_encoder = getattr(
                self, f"guidance_encoder_{self.guidance_types[guidance_idx]}"
            )
            guidance_fea = guidance_encoder(guidance_cond)
            guidance_fea_lst += [guidance_fea]

        guidance_fea = torch.stack(guidance_fea_lst, dim=0).sum(0)

        # if not uncond_fwd:
        #     ref_timesteps = torch.zeros_like(timesteps)
        #     self.reference_unet(
        #         ref_image_latents,
        #         ref_timesteps,
        #         encoder_hidden_states=clip_image_embeds,
        #         return_dict=False,
        #     )
        #     self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.reference_unet( # VF: we just want to use the reference unet 
            noisy_latents,
            timesteps,
            encoder_hidden_states= encoder_hidden_states,
            guidance_fea=guidance_fea,
            added_cond_kwargs= {
                "text_embeds": encoder_hidden_states , # You need to provide this
                "time_ids":timesteps
            },
            
       #     encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred