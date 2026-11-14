# Creates synthetic training data so you can test training TODAY
import json, os, random

os.makedirs("data/annotated", exist_ok=True)
random.seed(0)

ORGS = ["Acme Corp", "Beta LLC", "Gamma Inc", "Delta Co", "Epsilon Ltd"]
DATES = ["31 December 2024", "1 January 2025", "15 March 2024", "30 June 2025"]
MONEYS = ["$10,000", "$250,000", "€5,000", "$1,000,000"]

def make_entity(text, kind):
    v = random.choice({"ORG": ORGS, "DATE": DATES, "MONEY": MONEYS}[kind])
    start = text.find(v)
    return {"type": kind, "start": start, "end": start+len(v), "text": v}

rows = []
for _ in range(150):
    clause = random.choice(["Indemnification", "Limitation of Liability",
                            "Force Majeure", "Other"])
    if clause == "Indemnification":
        text = f"{random.choice(ORGS)} shall indemnify {random.choice(ORGS)} for all losses up to {random.choice(MONEYS)} incurred before {random.choice(DATES)}."
    elif clause == "Limitation of Liability":
        text = f"Neither party shall be liable for indirect damages exceeding {random.choice(MONEYS)} as of {random.choice(DATES)}."
    elif clause == "Force Majeure":
        text = f"In the event of force majeure, {random.choice(ORGS)} shall notify {random.choice(ORGS)} within 30 days from {random.choice(DATES)}."
    else:
        text = f"{random.choice(ORGS)} agrees to terms effective {random.choice(DATES)} with payment of {random.choice(MONEYS)}."
    entities = []
    for kind in ["ORG", "DATE", "MONEY"]:
        for v in {"ORG": ORGS, "DATE": DATES, "MONEY": MONEYS}[kind]:
            if v in text:
                start = text.find(v)
                entities.append({"type": kind, "start": start,
                                 "end": start+len(v), "text": v})
    rows.append({"text": text, "clause": clause, "entities": entities})

with open("data/annotated/train.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"Wrote {len(rows)} rows to data/annotated/train.jsonl")
