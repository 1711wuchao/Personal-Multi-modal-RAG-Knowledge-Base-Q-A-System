from pathlib import Path
import shutil
import tempfile
import unittest

from rag_app.config import AppConfig
from rag_app.evaluation import ragas_style_scores
from rag_app.pipeline import RAGPipeline


class FakeAnswerer:
    provider_name = "fake-llm"

    def generate(self, question: str, contexts: list[str]) -> str:
        return f"LLM回答：{question} -> {contexts[0][:8]}"


class RAGPipelineTests(unittest.TestCase):
    def test_markdown_ingest_query_and_incremental_skip(self):
        root = Path(tempfile.mkdtemp())
        try:
            docs = root / "docs"
            docs.mkdir()
            (docs / "agent.md").write_text(
                "# Agent 笔记\n\nRAG 系统包含离线索引、在线检索、BM25、多路召回和重排。",
                encoding="utf-8",
            )
            config = AppConfig.from_root(root)
            pipeline = RAGPipeline(config)

            first = pipeline.ingest_path(docs)
            second = pipeline.ingest_path(docs)
            answer = pipeline.answer("RAG 系统包含哪些环节？")

            self.assertEqual(first["indexed_files"], 1)
            self.assertEqual(second["skipped_files"], 1)
            self.assertGreaterEqual(answer["metrics"]["context_hit_rate"], 0.5)
            self.assertTrue(answer["sources"])
            self.assertIn("离线索引", answer["answer"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_evaluate_question_reports_ragas_style_scores(self):
        root = Path(tempfile.mkdtemp())
        try:
            doc = root / "knowledge.md"
            doc.write_text("OCR 用于提取图片文字，VLM 用于生成图片语义摘要。", encoding="utf-8")
            pipeline = RAGPipeline(AppConfig.from_root(root))
            pipeline.ingest_path(doc)

            result = pipeline.evaluate("OCR 和 VLM 分别做什么？", "OCR 提取文字，VLM 生成图片语义摘要")

            self.assertIn("answer_accuracy", result)
            self.assertIn("retrieval_quality", result)
            self.assertIn("context_hit_rate", result)
            self.assertGreater(result["retrieval_quality"], 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_semantic_paraphrase_keeps_faithfulness_score_useful(self):
        context = (
            "AI Agent 是围绕目标自主完成任务的智能应用，"
            "通常由大模型、工具调用、记忆系统、工作流和反馈机制组成。"
        )
        answer = (
            "AI Agent 可以理解为以大语言模型作为决策核心，"
            "结合外部工具、上下文记忆、流程编排和结果反馈来完成任务的应用。"
        )

        scores = ragas_style_scores("什么是 AI Agent？", answer, [context])

        self.assertGreaterEqual(scores["answer_accuracy"], 0.55)

    def test_changed_file_replaces_old_chunks(self):
        root = Path(tempfile.mkdtemp())
        try:
            doc = root / "note.md"
            doc.write_text("旧知识：系统只支持 Markdown。", encoding="utf-8")
            pipeline = RAGPipeline(AppConfig.from_root(root))
            pipeline.ingest_path(doc)

            doc.write_text("新知识：系统支持 PDF、Markdown、图片 OCR 和 VLM 摘要。", encoding="utf-8")
            result = pipeline.ingest_path(doc)
            answer = pipeline.answer("系统支持哪些内容？")

            self.assertEqual(result["indexed_files"], 1)
            self.assertEqual(result["total_chunks"], 1)
            self.assertIn("图片 OCR", answer["answer"])
            self.assertNotIn("只支持 Markdown", answer["answer"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_list_file_detail_and_delete_file(self):
        root = Path(tempfile.mkdtemp())
        try:
            pipeline = RAGPipeline(AppConfig.from_root(root))
            pipeline.ingest_upload("agent.md", "AI Agent 文件包含大模型、工具调用和记忆系统。".encode("utf-8"))
            pipeline.ingest_upload("ops.md", "运营助手文件包含数据报表、指标异常和运营建议。".encode("utf-8"))

            files = pipeline.list_files()
            agent_file = next(item for item in files if item["filename"] == "agent.md")
            detail = pipeline.file_detail(agent_file["id"])
            delete_result = pipeline.delete_file(agent_file["id"])
            answer = pipeline.answer("AI Agent 文件包含什么？")

            self.assertEqual(len(files), 2)
            self.assertEqual(detail["filename"], "agent.md")
            self.assertEqual(detail["chunks_count"], 1)
            self.assertIn("大模型", detail["chunks"][0]["preview"])
            self.assertEqual(delete_result["deleted_file"], "agent.md")
            self.assertFalse((pipeline.config.upload_dir / "agent.md").exists())
            self.assertNotIn("agent.md", [item["filename"] for item in pipeline.list_files()])
            self.assertNotIn("agent.md", [source["filename"] for source in answer["sources"]])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_ingest_invalidates_stale_empty_answer_cache(self):
        root = Path(tempfile.mkdtemp())
        try:
            doc = root / "cache.md"
            pipeline = RAGPipeline(AppConfig.from_root(root))

            before = pipeline.answer("RAG 如何提升准确率？")
            doc.write_text("RAG 通过 BM25 关键词检索、向量召回、多路召回和重排提升问答准确率。", encoding="utf-8")
            pipeline.ingest_path(doc)
            after = pipeline.answer("RAG 如何提升准确率？")

            self.assertFalse(before["sources"])
            self.assertTrue(after["sources"])
            self.assertFalse(after["cached"])
            self.assertIn("BM25", after["answer"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_answer_uses_configured_llm_answerer(self):
        root = Path(tempfile.mkdtemp())
        try:
            doc = root / "llm.md"
            doc.write_text("多模态 RAG 可以结合 OCR、BM25、向量召回和重排生成答案。", encoding="utf-8")
            pipeline = RAGPipeline(AppConfig.from_root(root), answerer=FakeAnswerer())
            pipeline.ingest_path(doc)

            answer = pipeline.answer("多模态 RAG 如何生成答案？")

            self.assertTrue(answer["answer"].startswith("LLM回答："))
            self.assertEqual(answer["answer_provider"], "fake-llm")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_reset_clears_knowledge_base_without_deleting_locked_chroma_files(self):
        root = Path(tempfile.mkdtemp())
        try:
            doc = root / "reset.md"
            doc.write_text("重置前的知识会进入 Chroma、manifest 和本地缓存。", encoding="utf-8")
            pipeline = RAGPipeline(AppConfig.from_root(root))
            pipeline.ingest_path(doc)
            pipeline.answer("重置前有哪些知识？")

            pipeline.reset()
            stats = pipeline.stats()
            answer = pipeline.answer("重置前有哪些知识？")

            self.assertEqual(stats["files"], 0)
            self.assertEqual(stats["chunks"], 0)
            self.assertEqual(stats["cache_files"], 0)
            self.assertFalse(answer["sources"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
