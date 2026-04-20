# add this to __main__.py temporarily
import torch
import numpy as np

model = Small_LLM_Model()

# encode a simple prompt
prompt_text = "Greet shrek"
prompt_ids = model.encode(prompt_text)
print(f"Token IDs: {prompt_ids}")

# get logits
input_tensor = torch.tensor([prompt_ids])
logits = model.get_logits_from_input_ids(input_tensor)

# we only care about the LAST token's logits
logits_np = logits[0, -1].detach().numpy().copy()

print(f"Logits shape: {logits_np.shape}")  # (151936,)
print(f"Max logit: {logits_np.max()}")
print(f"Min logit: {logits_np.min()}")

# what token does the LLM want most right now?
best_id = int(np.argmax(logits_np))
print(f"Best token ID: {best_id}")
print(f"Best token string: {id_to_str[best_id]}")
