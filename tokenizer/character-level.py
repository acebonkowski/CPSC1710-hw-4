# COPY PASTE for tools imported & seed initialization
import numpy as np
import tensorflow as tf

SEED = 1710
if SEED is not None:
    tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)


paragraphs = [
    "In 2026, we teach `intro-to-AI` with hands-on labs—no hype. Students ask: `Why tokens?` Because models read pieces, not words. E.g., `ChatGPT-5` ≠ `Chat`, `GPT`, `5` in all schemes. We track loss/accuracy, compare char/word/BPE, and test a URL: https://example.org/a/b?c=42. Café prices rose 3.7%—blame supply-chain weirdness (and ☕ demand).",
]

text = "\n".join(paragraphs)

print("Corpus length:", len(text)) # Output character length of glued together text
chars = sorted(list(set(text))) # Define list of different characters in the text & sort (new line > spaces > punctuation > capitalized letters > non-capitalized letters) w/o duplicates via set()
stoi = {c: i for i, c in enumerate(chars)} # Covert characters to numbers
itos = {i: c for c, i in stoi.items()} # Puts number-character-pairs into dictionary (= list of lists)
vocab_size = len(chars) # Output number of non-duplicate characters in text
print("Vocab:", chars) # Print list of sorted non-duplicate characters
print("Vocab Size:", len(chars))