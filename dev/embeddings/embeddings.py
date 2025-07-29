import os
import re
import gensim
from gensim.models import KeyedVectors

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.abspath(os.path.join(CURR_DIR, "../../"))
MODEL_FILE = os.path.join(CURR_DIR, "models", "ubercorpus.lowercased.lemmatized.word2vec.300d")
OUTPUT_FILE = os.path.join(CURR_DIR, "ukr-uum.embeddings.tsv")

command = f"cd {PATH} && lt-print -H uum-ukr.autobil.bin | hfst-txt2fst | hfst-invert | hfst-expand -c0"
output = os.popen(command).read()

ukrainian_words = []

for line in output.strip().split("\n"):
    if ":" in line:
        ukr = line.split(":")[0]
        word = re.sub(r"<.*?>", "", ukr)
        match = re.search(r"<[^>]+>", ukr)
        if match:
            first_tag = match.group(0)
            combined = f"{word}{first_tag}"
        else:
            combined = word
        combined = combined.strip()
        if combined and len(word) > 1:
            ukrainian_words.append(combined)

ukrainian_words = list(set(ukrainian_words))

print("loading model")
model = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=False)
print("loaded model")

threshold = 0.2
topn = 10

def helper(word):
    match = re.match(r"^(.*?)(<.*?>)?$", word)
    if match:
        base = match.group(1)
        tag = " " + match.group(2) if match.group(2) else ""
    else:
        base = word
        tag = ""
    spaced_word = " ".join(base)
    return spaced_word + tag, tag

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for word in ukrainian_words:
        base_word = re.sub(r"<.*?>", "", word)
        if base_word not in model.key_to_index:
            continue
        for sim_word, score in model.most_similar(base_word, topn=topn):
            if score < threshold or len(sim_word) <= 1:
                continue
            spaced_word, tag_word = helper(word)
            spaced_candidate, tag_candidate = helper(sim_word)
            if tag_word == "<num>" or tag_candidate == "<num>":
                continue
            f.write(f"{spaced_candidate}:{spaced_word}\t{score}\n")
