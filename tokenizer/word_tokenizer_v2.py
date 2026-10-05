paragraphs = [
    "In 2026, we teach `intro-to-AI` with hands-on labs—no hype. Students ask: `Why tokens?` Because models read pieces, not words. E.g., `ChatGPT-5` ≠ `Chat`, `GPT`, `5` in all schemes. We track loss/accuracy, compare char/word/BPE, and test a URL: https://example.org/a/b?c=42. Café prices rose 3.7%—blame supply-chain weirdness (and ☕ demand).",
]

text = "\n".join(paragraphs)


# 1) Convert the Text
text_lowercase = text.lower() # Make text lowercase
text_clean = "".join(c if c.isalnum() or c.isspace() else " " for c in text_lowercase) # Replace every character that is not a letter, digit or whitespace with a space
words = text_clean.split() # Split by whitespace & keep duplicate words
print("Total word count:", len(words))
tokens = list(dict.fromkeys(words)) # Split by whitespace & remove duplicate words (keeps order of first appearance)
print("Number of different words:", len(tokens))


# 2) Create a Vocabulary
vocabulary = sorted(set(tokens)) # Sorted list of unique words
stoi = {w: i for i, w in enumerate(vocabulary)} # Convert words to numbers
itos = {i: w for w, i in stoi.items()} # Convert numbers back to words
vocabulary_size = len(vocabulary) # Number of different words in vocabulary
print("Vocabulary:", stoi.items()) # Print word-number-pairs
print("Vocabulary Size:", vocabulary_size)


# 3) Encode Tokens into IDs
encoded_tokens = [stoi[w] for w in tokens] # Get the ID of every word in tokens
print("Encoded Tokens:", encoded_tokens)


# 4) Decode IDs into Tokens
decoded_tokens = [itos[i] for i in encoded_tokens] # Get the word of every ID in encoded_tokens
print("Decoded Tokens:", decoded_tokens)


# 5) Generate 5-Word Sentence
sentence = " ".join(decoded_tokens[:5]) # Glue first 5 words together with a space in between
print("Sentence:", sentence)
