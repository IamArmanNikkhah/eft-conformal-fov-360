# models/peft_wrapper.py

import torch
import torch.nn as nn


class LoRA_Adapter(nn.Module):
    def __init__(self, base_model, rank=4):
        super().__init__()
        self.base_model = base_model

        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Define the Adapter
        hidden_dim = self.base_model.input_proj.out_features  # = d_model (256 in model.py)
        self.lora_A = nn.Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.lora_B_pref = nn.Parameter(torch.randn(rank, 2) * 0.01)  # (Yaw, Pitch) for prefetch
        self.lora_B_dead = nn.Parameter(torch.randn(rank, 2) * 0.01)  # (Yaw, Pitch) for deadline

    def forward(self, x):
        # Run the frozen base model
        base_pref, base_dead = self.base_model(x)

        #Run the internal logic to get the hidden state before the final head
        embedded = self.base_model.input_proj(x)
        embedded = self.base_model.pos_encoder(embedded)
        trans_out = self.base_model.encoder(embedded)[:, -1, :]  # last token (batch, hidden_dim)

        # Calculate Adapter adjustment
        adapter_pref = trans_out @ self.lora_A @ self.lora_B_pref
        adapter_dead = trans_out @ self.lora_A @ self.lora_B_dead

        # Result = Base Knowledge + Personal Adjustment
        return base_pref + adapter_pref, base_dead + adapter_dead

#Test the LoRA_Adapter with zeros input
if __name__ == "__main__":
    from model import PooledFoVTransformer

    base_model = PooledFoVTransformer()
    personalized_model = LoRA_Adapter(base_model, rank=4)

    x = torch.zeros(4, 15, 2)
    y_pref, y_dead = personalized_model(x)
    print("prefetch:", y_pref.shape)
    print("deadline:", y_dead.shape)
