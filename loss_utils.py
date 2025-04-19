import torch
import torch.nn.functional as F
from typing import List

def get_batch_logps(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    temperature: float = 1.0
) -> torch.FloatTensor:
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits / temperature  # [B, L, V]
    
    # log-softmax over vocab
    log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]
    
    # causal shift: predict token t from tokens < t
    shift_log_probs = log_probs[:, :-1, :]     # [B, L-1, V]
    shift_labels    = input_ids[:, 1:]         # [B, L-1]
    
    # gather log-prob of the true token
    token_logps = shift_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    
    # mask out padding-label positions
    label_mask = attention_mask[:, 1:] == 1    # [B, L-1]
    token_logps = token_logps * label_mask
    
    return token_logps

def unpad_logps(
    token_logps: torch.FloatTensor,
    attention_mask: torch.LongTensor
) -> List[List[float]]:
    # compute valid lengths: number of real tokens minus 1 for shift
    lengths = (attention_mask.sum(dim=1) - 1).tolist()
    return [
        token_logps[i, :lengths[i]].cpu().tolist()
        for i in range(token_logps.size(0))
    ]
