import os
import sys
import torch
import argparse


class LatentEditor:
    def __init__(self):
        self.interfacegan_directions = {
                "age": "editings/interfacegan_directions/age.pt",
                "smile": "editings/interfacegan_directions/smile.pt",
                "rotation": "editings/interfacegan_directions/rotation.pt",
            }

        self.interfacegan_directions_tensors = {
            name: torch.load(path).cuda()
            for name, path in self.interfacegan_directions.items()
        }

    def get_single_interface_gan_edits_with_direction(
        self, start_w, factors, direction
    ):
        latents_to_display = []
        for factor in factors:
            latents_to_display.append(
                self.apply_interfacegan(
                    start_w, self.interfacegan_directions_tensors[direction], factor / 2
                )
            )
        return latents_to_display
    
    def apply_interfacegan(self, latent, direction, factor=1, factor_range=None):
        edit_latents = []
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latent + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = latent + factor * direction
        return edit_latents