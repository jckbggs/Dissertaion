import time
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import faiss
import pynvml

# 1. GPU
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))

# 2. GPT-2 Small
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda").eval()
print("GPT-2 params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# 3. WikiText-103 (downloads the dataset)
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
print("WikiText-103 validation rows:", len(ds))

# 4. FAISS k-means
x = np.random.randn(10000, 64).astype("float32")
km = faiss.Kmeans(64, 16, niter=5)
km.train(x)
print("FAISS k-means OK")

# 5. NVML power and energy
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print("NVML power now (W):", pynvml.nvmlDeviceGetPowerUsage(h) / 1000)

ids = tok("The quick brown fox", return_tensors="pt").input_ids.to("cuda")

# warm-up so CUDA start-up cost is not measured
with torch.no_grad():
    model.generate(ids, max_new_tokens=20, do_sample=False)
torch.cuda.synchronize()

try:
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)  # millijoules
except pynvml.NVMLError as e:
    e0 = None
    print("Total-energy counter not supported on this GPU:", e)

t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=200, do_sample=False)
torch.cuda.synchronize()
dt = time.time() - t0

if e0 is not None:
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    print(f"NVML energy: {(e1 - e0) / 1000:.2f} J over {dt:.2f} s")
print("Generated:", tok.decode(out[0])[:80])

# 6. CodeCarbon (optional, may fail on Windows)
try:
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker(project_name="env_check", save_to_file=False, log_level="error")
    tracker.start()
    with torch.no_grad():
        model.generate(ids, max_new_tokens=200, do_sample=False)
    tracker.stop()
    print("CodeCarbon energy (kWh):", tracker.final_emissions_data.energy_consumed)
except Exception as e:
    print("CodeCarbon failed:", type(e).__name__, e)