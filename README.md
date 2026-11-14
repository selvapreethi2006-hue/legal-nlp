# Legal Clause Classification & Entity Extraction

Deterministic NLP pipeline for OCR-processed legal contracts — satisfies task 3.md requirements with zero hallucination.

## Architecture

| Stage | Implementation | File |
|-------|----------------|------|
| Preprocessing | OCR normalization, unicode cleanup | `src/preprocess.py` |
| Tokenization | BERT WordPiece with offset mapping | `src/preprocess.py` |
| Embeddings | BERT-base-uncased (768-dim) | `src/model.py` |
| NER | BERT → BiLSTM → CRF (7 BIO tags) | `src/model.py` |
| Classifier | BERT + class-weighted softmax (4 classes) | `src/train_clause.py` |
| Inference | JSON output + precision threshold | `src/predict_v2.py` |
| Evaluation | seqeval + sklearn + noise robustness | `src/evaluate.py` |

## Zero Hallucination Guarantee

```python
# In predict_v2.py
assert ent["text"] == text[ent["start"]:ent["end"]]