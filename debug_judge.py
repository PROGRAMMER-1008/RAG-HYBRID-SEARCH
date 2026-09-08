import sys
sys.path.insert(0, '.')
from app.generation.prompt import GenerationPrompt
from app.generation.llm_client import generate

source = """### Focal Tversky Loss
A custom loss function for segmentation that:
- Handles class imbalance (tumors are small vs. whole image)
- Penalizes false negatives more heavily (missing a tumor is worse than a false alarm)"""

STRICT_SYS = """You are a strict fact-checker. You will be given ONE claim and ONE source passage. Your only job is to decide whether the source passage actually supports the claim.

Respond with EXACTLY one word: SUPPORTED, UNSUPPORTED, or PARTIAL.
- SUPPORTED: the passage directly and clearly supports the claim.
- PARTIAL: the passage is related but doesn't fully support the specific detail claimed (e.g. right topic, wrong number).
- UNSUPPORTED: the passage does not support the claim at all.

Respond with only that one word, nothing else."""

# Claim WITH the trailing citation marker still attached, exactly as
# the real pipeline sends it
claim_with_marker = "The custom loss function used for segmentation, Focal Tversky Loss, handles class imbalance [1]."

prompt = GenerationPrompt(
    system_prompt=STRICT_SYS,
    user_prompt=f"Claim: {claim_with_marker}\n\nSource passage:\n{source}\n\nVerdict:",
    numbered_chunks=[],
)

print("=== Claim WITH trailing [1]. marker (matches real pipeline exactly), 3 runs ===")
for i in range(3):
    print(f"  run {i+1}:", repr(generate(prompt, provider="groq")))