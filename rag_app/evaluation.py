from __future__ import annotations

import math

from .text_utils import tokenize


CONCEPT_ALIASES = [
    {"aiagent", "agent", "智能体", "智能应用"},
    {"llm", "大模型", "大语言模型", "模型"},
    {"toolcalling", "functioncalling", "工具调用", "调用工具", "外部工具", "工具"},
    {"memory", "记忆系统", "上下文记忆", "长期记忆", "短期记忆", "记忆"},
    {"workflow", "工作流", "流程编排", "编排", "流程"},
    {"feedback", "反馈机制", "结果反馈", "反馈", "持续调整"},
    {"rag", "知识检索", "检索增强", "知识库", "上下文"},
    {"bm25", "关键词检索", "关键词召回"},
    {"embedding", "向量检索", "向量召回", "语义检索"},
    {"rerank", "重排", "排序"},
    {"ocr", "文字提取", "文本提取"},
    {"vlm", "视觉模型", "图片语义"},
]


def compact_text(text: str) -> str:
    return "".join(tokenize(text))


def overlap_score(left: str, right: str) -> float:
    a = set(tokenize(left))
    b = set(tokenize(right))
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 4)


def coverage_score(query: str, context: str) -> float:
    query_tokens = set(tokenize(query))
    context_tokens = set(tokenize(context))
    if not query_tokens:
        return 0.0
    return round(len(query_tokens & context_tokens) / len(query_tokens), 4)


def char_ngrams(text: str, size: int = 2) -> set[str]:
    compact = compact_text(text)
    if not compact:
        return set()
    if len(compact) <= size:
        return {compact}
    return {compact[idx : idx + size] for idx in range(len(compact) - size + 1)}


def cosine_score(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for token in left_tokens:
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in right_tokens:
        right_counts[token] = right_counts.get(token, 0) + 1
    common = set(left_counts) & set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_overlap_score(left: str, right: str) -> float:
    token_overlap = overlap_score(left, right)
    token_cosine = cosine_score(left, right)
    left_grams = char_ngrams(left)
    right_grams = char_ngrams(right)
    gram_overlap = 0.0
    if left_grams and right_grams:
        gram_overlap = len(left_grams & right_grams) / len(left_grams | right_grams)
    return round(max(token_overlap, token_cosine, gram_overlap), 4)


def concept_hits(text: str) -> set[int]:
    compact = compact_text(text)
    hits: set[int] = set()
    for idx, aliases in enumerate(CONCEPT_ALIASES):
        if any(alias.lower().replace(" ", "") in compact for alias in aliases):
            hits.add(idx)
    return hits


def concept_overlap_score(left: str, right: str) -> float:
    left_hits = concept_hits(left)
    right_hits = concept_hits(right)
    if not left_hits or not right_hits:
        return 0.0
    return round(len(left_hits & right_hits) / len(left_hits | right_hits), 4)


def answer_faithfulness_score(question: str, answer: str, contexts: list[str], ground_truth: str = "") -> float:
    reference = ground_truth or "\n".join(contexts)
    if not answer.strip() or not reference.strip():
        return 0.0
    semantic_similarity = semantic_overlap_score(answer, reference)
    concept_similarity = concept_overlap_score(answer, reference)
    answer_context_coverage = coverage_score(answer, reference)
    question_context_coverage = coverage_score(question, reference)
    score = (
        0.35 * max(semantic_similarity, concept_similarity)
        + 0.25 * concept_similarity
        + 0.25 * answer_context_coverage
        + 0.15 * question_context_coverage
    )
    return round(min(score, 1.0), 4)


def ragas_style_scores(question: str, answer: str, contexts: list[str], ground_truth: str = "") -> dict[str, float]:
    joined_context = "\n".join(contexts)
    context_hit_rate = coverage_score(question, joined_context)
    retrieval_quality = max((coverage_score(question, ctx) for ctx in contexts), default=0.0)
    answer_accuracy = answer_faithfulness_score(question, answer, contexts, ground_truth)
    return {
        "answer_accuracy": answer_accuracy,
        "retrieval_quality": retrieval_quality,
        "context_hit_rate": context_hit_rate,
    }
