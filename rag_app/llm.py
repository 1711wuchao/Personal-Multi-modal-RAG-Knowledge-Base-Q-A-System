from __future__ import annotations

from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import AppConfig


class Answerer(Protocol):
    provider_name: str

    def generate(self, question: str, contexts: list[str]) -> str:
        ...


class LangChainAnswerer:
    provider_name = "langchain-openai"

    def __init__(self, config: AppConfig):
        self.chat = ChatOpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            temperature=0.2,
            timeout=20,
            max_retries=1,
        )

    def generate(self, question: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(f"[{idx}] {text}" for idx, text in enumerate(contexts[:5], start=1))
        messages = [
            SystemMessage(
                content=(
                    "你是个人多模态 RAG 知识库问答助手。"
                    "必须只依据给定上下文回答，回答要简洁、准确，并指出关键信息来自检索内容。"
                    "如果上下文不足，要明确说明缺少资料。"
                )
            ),
            HumanMessage(content=f"问题：{question}\n\n检索上下文：\n{context_text}"),
        ]
        response = self.chat.invoke(messages)
        return str(response.content).strip()


def build_answerer(config: AppConfig) -> Answerer | None:
    if not config.llm_api_key:
        return None
    return LangChainAnswerer(config)
