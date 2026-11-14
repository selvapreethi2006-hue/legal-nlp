# setup_project.py — run this ONCE to create all project files
import os

FILES = {}

FILES["src/preprocess.py"] = '''import re
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def clean_ocr(text: str) -> str:
    text = re.sub(r"\\s+", " ", text)
    text = text.replace("\\u2019", "'").replace("\\u2018", "'")
    text = text.replace("\\u201c", '"').replace("\\u201d", '"')
    return text.strip()

def tokenize_with_offsets(text: str, max_len: int = 512):
    enc = tokenizer(
        text, return_offsets_mapping=True, truncation=True,
        max_length=max_len, padding=False,
    )
    return enc["input_ids"], enc["offset_mapping"]
'''

FILES["src/dataset.py"] = '''import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

NER_LABELS = ["O", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "B-MONEY", "I-MONEY"]
NER2ID = {l: i for i, l in enumerate(NER_LABELS)}
CLAUSE_LABELS = ["Indemnification", "Limitation of Liability",
                 "Force Majeure", "Other"]
CLAUSE2ID = {l: i for i, l in enumerate(CLAUSE_LABELS)}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def chars_to_ner_labels(text, entities, offsets):
    labels = []
    span_map = [(e["start"], e["end"], e["type"]) for e in entities]
    for (s, e) in offsets:
        if s == e:
            labels.append(-100)
            continue
        tag = "O"
        for (es, ee, etype) in span_map:
            if s >= es and e <= ee:
                tag = ("B-" if s == es else "I-") + etype
                break
            if s < ee and e > es:
                tag = "I-" + etype
                break
        labels.append(NER2ID[tag])
    return labels

class NERDataset(Dataset):
    def __init__(self, path, max_len=256):
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                enc = tokenizer(rec["text"], return_offsets_mapping=True,
                                truncation=True, max_length=max_len,
                                padding="max_length")
                labels = chars_to_ner_labels(rec["text"], rec["entities"],
                                             enc["offset_mapping"])
                self.items.append({
                    "input_ids": torch.tensor(enc["input_ids"]),
                    "attention_mask": torch.tensor(enc["attention_mask"]),
                    "labels": torch.tensor(labels),
                })
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class ClauseDataset(Dataset):
    def __init__(self, path, max_len=256):
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                enc = tokenizer(rec["text"], truncation=True,
                                max_length=max_len, padding="max_length")
                self.items.append({
                    "input_ids": torch.tensor(enc["input_ids"]),
                    "attention_mask": torch.tensor(enc["attention_mask"]),
                    "labels": torch.tensor(CLAUSE2ID[rec["clause"]]),
                })
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]
'''

FILES["src/model.py"] = '''import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from torchcrf import CRF

class BertBiLSTMCRF(nn.Module):
    def __init__(self, num_labels, lstm_hidden=256, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden = self.bert.config.hidden_size
        self.bilstm = nn.LSTM(hidden, lstm_hidden // 2, num_layers=1,
                              bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)
        seq, _ = self.bilstm(seq)
        emissions = self.classifier(seq)
        if labels is not None:
            mask = attention_mask.bool()
            clean = labels.clone()
            clean[clean == -100] = 0
            loss = -self.crf(emissions, clean, mask=mask, reduction="mean")
            return loss
        return self.crf.decode(emissions, mask=attention_mask.bool())

def build_clause_model(num_labels):
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels)
'''

FILES["src/train_ner.py"] = '''import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import NERDataset, NER_LABELS
from model import BertBiLSTMCRF

def train(data_path, epochs=5, batch_size=8, lr=2e-5):
    ds = NERDataset(data_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = BertBiLSTMCRF(num_labels=len(NER_LABELS))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        total = 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {ep+1} loss = {total/len(dl):.4f}")
    torch.save(model.state_dict(), "ner_model.pt")
    print("Saved ner_model.pt")

if __name__ == "__main__":
    train("data/annotated/train.jsonl")
'''

FILES["src/train_clause.py"] = '''import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import ClauseDataset, CLAUSE_LABELS
from model import build_clause_model

def train(data_path, epochs=3, batch_size=8, lr=2e-5):
    ds = ClauseDataset(data_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = build_clause_model(len(CLAUSE_LABELS))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train(); total = 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward(); opt.step(); opt.zero_grad()
            total += out.loss.item()
        print(f"Epoch {ep+1} loss = {total/len(dl):.4f}")
    model.save_pretrained("clause_model")
    print("Saved clause_model/")

if __name__ == "__main__":
    train("data/annotated/train.jsonl")
'''

FILES["src/predict.py"] = '''import json, torch, sys
from preprocess import clean_ocr, tokenize_with_offsets
from dataset import NER_LABELS
from model import BertBiLSTMCRF

def predict_entities(text, ner_model_path="ner_model.pt"):
    text = clean_ocr(text)
    ids, offsets = tokenize_with_offsets(text)
    model = BertBiLSTMCRF(num_labels=len(NER_LABELS))
    model.load_state_dict(torch.load(ner_model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        preds = model(input_ids=torch.tensor([ids]),
                      attention_mask=torch.tensor([[1]*len(ids)]))[0]
    entities, cur = [], None
    for (s, e), p in zip(offsets, preds):
        if s == e: continue
        lab = NER_LABELS[p]
        if lab.startswith("B-"):
            if cur: entities.append(cur)
            cur = {"type": lab[2:], "start": s, "end": e, "text": text[s:e]}
        elif lab.startswith("I-") and cur and cur["type"] == lab[2:]:
            cur["end"] = e; cur["text"] = text[cur["start"]:e]
        else:
            if cur: entities.append(cur)
            cur = None
    if cur: entities.append(cur)
    for ent in entities:
        assert ent["text"] == text[ent["start"]:ent["end"]], "span mismatch!"
    return entities

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Acme Corp shall indemnify Beta LLC."
    print(json.dumps({"text": text, "entities": predict_entities(text)},
                     indent=2, ensure_ascii=False))
'''

FILES["make_sample_data.py"] = '''# Creates synthetic training data so you can test training TODAY
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
        f.write(json.dumps(r, ensure_ascii=False)+"\\n")
print(f"Wrote {len(rows)} rows to data/annotated/train.jsonl")
'''

for path, content in FILES.items():
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created {path}")

print("\nDone. Now run:")
print("  python make_sample_data.py")
print("  cd src && python train_ner.py")