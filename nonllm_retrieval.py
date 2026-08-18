#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Non-LLM retrieval baselines for G4.

The baseline transfers labels from nearest training instances to target
instances. Text representations are built from the same event/unit payloads
used by the LLM prompt path, then neighbors vote for aligned and contradictory
human-value labels.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

InstanceId = Tuple[str, str, str, str]
GoldDict = Dict[InstanceId, Dict[str, Set[str]]]


def _clip(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(_clip(x, max_chars) for x in text)
    elif isinstance(text, dict):
        text = " ".join(_clip(v, max_chars) for v in text.values())
    else:
        text = str(text)
    text = " ".join(text.split())
    return text[:max_chars] if max_chars and max_chars > 0 else text


def _payload_text(
    *,
    data_utils,
    events_by_guid: Dict[str, Dict[str, Any]],
    iid: InstanceId,
    label_space: str,
    hv_label: str,
    hv_label_desc: str,
    max_chars_per_field: int,
    include_event_content_for_article: bool,
) -> str:
    guid, unit_level, _unit_id, _actor = iid
    event = events_by_guid.get(guid)
    if event is None:
        return " ".join([unit_level, guid])

    payload = data_utils.build_input_payload(
        event,
        iid,
        label_space,
        hv_label,
        hv_label_desc,
    )

    fields = [
        payload.get("actor", ""),
        payload.get("unit_level", ""),
        payload.get("unit_title", ""),
        payload.get("unit_text", ""),
        payload.get("evidence_sentences", ""),
        payload.get("context_sentences", ""),
    ]

    if include_event_content_for_article or unit_level != "article":
        fields.append(payload.get("article_title", ""))

    return "\n".join(_clip(x, max_chars_per_field) for x in fields if _clip(x, max_chars_per_field))


def _make_records(
    *,
    events_by_guid: Dict[str, Dict[str, Any]],
    gold: GoldDict,
    data_utils,
    label_space: str,
    hv_label: str,
    hv_label_desc: str,
    max_chars_per_field: int,
    include_event_content_for_article: bool,
) -> Tuple[List[InstanceId], List[str]]:
    iids = list(gold.keys())
    texts = [
        _payload_text(
            data_utils=data_utils,
            events_by_guid=events_by_guid,
            iid=iid,
            label_space=label_space,
            hv_label=hv_label,
            hv_label_desc=hv_label_desc,
            max_chars_per_field=max_chars_per_field,
            include_event_content_for_article=include_event_content_for_article,
        )
        for iid in iids
    ]
    return iids, texts


def _tfidf_similarity(
    train_texts: Sequence[str],
    target_texts: Sequence[str],
    *,
    tfidf_ngram_max: int,
    tfidf_min_df: int,
    tfidf_max_df: float,
    tfidf_max_features: int,
) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    max_features: Optional[int] = tfidf_max_features if tfidf_max_features and tfidf_max_features > 0 else None
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, tfidf_ngram_max),
        min_df=tfidf_min_df,
        max_df=tfidf_max_df,
        max_features=max_features,
        norm="l2",
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    target_matrix = vectorizer.transform(target_texts)
    return cosine_similarity(target_matrix, train_matrix)


def _sbert_similarity(
    train_texts: Sequence[str],
    target_texts: Sequence[str],
    *,
    sbert_model_name: str,
    sbert_batch_size: int,
    sbert_device: str,
    sbert_cache_folder: str,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    kwargs: Dict[str, Any] = {}
    if sbert_device:
        kwargs["device"] = sbert_device
    if sbert_cache_folder:
        kwargs["cache_folder"] = sbert_cache_folder

    model = SentenceTransformer(sbert_model_name, **kwargs)
    train_emb = model.encode(
        list(train_texts),
        batch_size=sbert_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    target_emb = model.encode(
        list(target_texts),
        batch_size=sbert_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.matmul(np.asarray(target_emb), np.asarray(train_emb).T)


def _top_neighbors(
    scores: np.ndarray,
    *,
    target_iid: InstanceId,
    train_iids: Sequence[InstanceId],
    k: int,
    min_sim: float,
    level_filter: bool,
) -> List[Tuple[int, float]]:
    order = np.argsort(-scores)
    out: List[Tuple[int, float]] = []
    for idx in order:
        sim = float(scores[idx])
        if sim < min_sim:
            continue
        if level_filter and train_iids[idx][1] != target_iid[1]:
            continue
        out.append((int(idx), sim))
        if len(out) >= k:
            break
    return out


def _resolve_conflicts(
    aligned_scores: Dict[str, float],
    contra_scores: Dict[str, float],
    *,
    conflict_mode: str,
) -> Tuple[Set[str], Set[str]]:
    aligned = set(aligned_scores)
    contra = set(contra_scores)
    conflicts = aligned & contra

    for label in conflicts:
        if conflict_mode == "drop":
            aligned.discard(label)
            contra.discard(label)
        elif conflict_mode == "prefer_aligned":
            contra.discard(label)
        elif conflict_mode == "prefer_contradictory":
            aligned.discard(label)
        else:
            if aligned_scores[label] >= contra_scores[label]:
                contra.discard(label)
            else:
                aligned.discard(label)
    return aligned, contra


def _cap_labels(labels: Iterable[str], scores: Dict[str, float], top_m: int) -> Set[str]:
    labels = set(labels)
    if top_m and top_m > 0:
        return set(sorted(labels, key=lambda x: (-scores.get(x, 0.0), x))[:top_m])
    return labels


def _vote(
    neighbors: List[Tuple[int, float]],
    *,
    train_iids: Sequence[InstanceId],
    train_gold: GoldDict,
    vote_threshold: float,
    top_m_aligned: int,
    top_m_contra: int,
    conflict_mode: str,
) -> Dict[str, Set[str]]:
    aligned_scores: Dict[str, float] = defaultdict(float)
    contra_scores: Dict[str, float] = defaultdict(float)
    total_weight = sum(max(sim, 0.0) for _, sim in neighbors) or 1.0

    for idx, sim in neighbors:
        weight = max(float(sim), 0.0) / total_weight
        labels = train_gold.get(train_iids[idx], {})
        for label in labels.get("aligned", set()):
            aligned_scores[str(label)] += weight
        for label in labels.get("contradictory", set()):
            contra_scores[str(label)] += weight

    aligned_scores = {k: v for k, v in aligned_scores.items() if v >= vote_threshold}
    contra_scores = {k: v for k, v in contra_scores.items() if v >= vote_threshold}

    aligned, contra = _resolve_conflicts(
        aligned_scores,
        contra_scores,
        conflict_mode=conflict_mode,
    )
    aligned = _cap_labels(aligned, aligned_scores, top_m_aligned)
    contra = _cap_labels(contra, contra_scores, top_m_contra)

    return {
        "aligned": aligned,
        "contradictory": contra,
    }


def run_retrieval_baseline(
    *,
    method: str,
    train_events_by_guid: Dict[str, Dict[str, Any]],
    train_gold: GoldDict,
    target_events_by_guid: Dict[str, Dict[str, Any]],
    target_gold: GoldDict,
    label_space: str,
    hv_label: str,
    hv_label_desc: str,
    data_utils,
    level_filter: bool = True,
    k: int = 5,
    vote_threshold: float = 0.3,
    min_sim: float = 0.0,
    top_m_aligned: int = 0,
    top_m_contra: int = 0,
    conflict_mode: str = "prefer_higher",
    max_chars_per_field: int = 6000,
    include_event_content_for_article: bool = False,
    tfidf_ngram_max: int = 2,
    tfidf_min_df: int = 2,
    tfidf_max_df: float = 0.95,
    tfidf_max_features: int = 200000,
    sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    sbert_batch_size: int = 32,
    sbert_device: str = "",
    sbert_cache_folder: str = "",
) -> GoldDict:
    train_iids, train_texts = _make_records(
        events_by_guid=train_events_by_guid,
        gold=train_gold,
        data_utils=data_utils,
        label_space=label_space,
        hv_label=hv_label,
        hv_label_desc=hv_label_desc,
        max_chars_per_field=max_chars_per_field,
        include_event_content_for_article=include_event_content_for_article,
    )
    target_iids, target_texts = _make_records(
        events_by_guid=target_events_by_guid,
        gold=target_gold,
        data_utils=data_utils,
        label_space=label_space,
        hv_label=hv_label,
        hv_label_desc=hv_label_desc,
        max_chars_per_field=max_chars_per_field,
        include_event_content_for_article=include_event_content_for_article,
    )

    if method == "tfidf":
        similarity = _tfidf_similarity(
            train_texts,
            target_texts,
            tfidf_ngram_max=tfidf_ngram_max,
            tfidf_min_df=tfidf_min_df,
            tfidf_max_df=tfidf_max_df,
            tfidf_max_features=tfidf_max_features,
        )
    elif method == "sbert":
        similarity = _sbert_similarity(
            train_texts,
            target_texts,
            sbert_model_name=sbert_model_name,
            sbert_batch_size=sbert_batch_size,
            sbert_device=sbert_device,
            sbert_cache_folder=sbert_cache_folder,
        )
    else:
        raise ValueError(f"Unsupported retrieval method: {method}")

    predictions: GoldDict = {}
    for row_idx, target_iid in enumerate(target_iids):
        neighbors = _top_neighbors(
            similarity[row_idx],
            target_iid=target_iid,
            train_iids=train_iids,
            k=k,
            min_sim=min_sim,
            level_filter=level_filter,
        )
        predictions[target_iid] = _vote(
            neighbors,
            train_iids=train_iids,
            train_gold=train_gold,
            vote_threshold=vote_threshold,
            top_m_aligned=top_m_aligned,
            top_m_contra=top_m_contra,
            conflict_mode=conflict_mode,
        )

    return predictions
