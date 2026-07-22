"""MultipleNegativesRankingLoss로 (query, 정답문서) 쌍 대조학습 (in-batch negative)."""

from __future__ import annotations

from typing import Dict, Iterable, List

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def build_training_examples(
    train_qids: Iterable[str],
    train_gold: Dict[str, Dict[str, float]],
    queries: Dict[str, str],
    corpus_text: Dict[str, str],
    query_prefix: str = "",
    passage_prefix: str = "",
) -> List[InputExample]:
    examples = []
    for qid in train_qids:
        for cid, score in train_gold.get(qid, {}).items():
            if score > 0 and cid in corpus_text:
                examples.append(
                    InputExample(
                        texts=[query_prefix + queries[qid], passage_prefix + corpus_text[cid]]
                    )
                )
    return examples


def finetune(
    model: SentenceTransformer,
    examples: List[InputExample],
    epochs: int = 1,
    batch_size: int = 32,
) -> SentenceTransformer:
    """model을 in-place로 파인튜닝하고 그대로 반환한다."""
    train_dl = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(train_dl, train_loss)],
        epochs=epochs,
        warmup_steps=max(1, int(0.1 * len(train_dl))),
        show_progress_bar=True,
    )
    return model
