"""Extract a short, detector-friendly noun phrase from a stage description."""
import re
VERBS = r"(walk|go|head|move|step|proceed|continue|exit|leave|enter|come|get|turn|stop|wait|stand|face|pass|cross|climb|take|make|keep|follow|reach|arrive)"
ADV = r"(straight|forward|ahead|directly|slowly|quickly|clockwise|counter-?clockwise|around|past|along|toward|towards|down|up|over|across|through|out|back|into|onto|in|to|at|by|near|beside|next|of|the|a|an|your|then|and|until|you|there|here|left|right|on|from|off|behind|under|beneath|inside|outside|front|end|side|middle|way|other)"
def landmark_np(desc: str):
    d = desc.lower().strip().rstrip(".!? ")
    d = re.sub(r"\b(on|to)\s+(your|the)\s+(left|right)\b", " ", d)
    d = re.sub(r"\b(the|a)\s+(first|second|third|next|last)\b", " ", d)
    words = re.findall(r"[a-z][a-z-]*", d)
    # drop leading verbs/adverbs/function words
    i = 0
    while i < len(words) and (re.fullmatch(VERBS, words[i]) or re.fullmatch(ADV, words[i])):
        i += 1
    words = words[i:]
    if not words:
        return None
    # cut at the first preposition/conjunction after the head
    out = []
    for w in words:
        if out and re.fullmatch(r"(to|of|on|in|at|into|toward|towards|with|and|that|which|where|until|near|beside|by|from|through|past|for|behind|under|before|after)", w):
            break
        if re.fullmatch(r"(the|a|an|your|then|you|it|is|are)", w):
            continue
        out.append(w)
        if len(out) >= 3:
            break
    return " ".join(out) if out else None

if __name__ == "__main__":
    for t in ["Walk straight down the hallway", "Walk out of the bedroom into the hall", "Stop beside the large door", "Turn left", "Go up the stairs",
              "Walk to the far end of the lobby", "Wait under the exit sign", "Enter the kitchen", "Walk past the kitchen counter and windows",
              "Go forward down hallway until you reach flowers", "Stop after stepping on the rug in the hallway", "Walk clockwise around the bed"]:
        print(f"{t:52s} -> {landmark_np(t)}")
