import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Brain,
  Database,
  FileSearch,
  FileText,
  Gauge,
  Image,
  Loader2,
  MessageSquareText,
  Trash2,
  RefreshCw,
  Search,
  UploadCloud
} from "lucide-react";
import {
  ask,
  deleteKnowledgeFile,
  fetchFileDetail,
  fetchFiles,
  fetchStats,
  uploadDocument,
  type AskResponse,
  type KnowledgeFile,
  type KnowledgeFileDetail,
  type Stats
} from "./api";
import "./styles.css";

function App() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [files, setFiles] = useState<KnowledgeFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<KnowledgeFileDetail | null>(null);
  const [question, setQuestion] = useState("RAG 系统如何提升问答准确率？");
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [message, setMessage] = useState("");

  const loadWorkspace = async () => {
    const [nextStats, nextFiles] = await Promise.all([fetchStats(), fetchFiles()]);
    setStats(nextStats);
    setFiles(nextFiles.files);
    if (selectedFile && !nextFiles.files.some((file) => file.id === selectedFile.id)) {
      setSelectedFile(null);
    }
  };

  useEffect(() => {
    loadWorkspace().catch(() => setMessage("后端服务未启动，请先运行 uvicorn rag_app.api:app --port 8010"));
  }, []);

  const metrics = useMemo(() => {
    if (!answer) return [];
    return [
      ["答案忠实度", answer.metrics.answer_accuracy],
      ["检索相关性", answer.metrics.retrieval_quality],
      ["问题覆盖率", answer.metrics.context_hit_rate]
    ];
  }, [answer]);

  const submitQuestion = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setMessage("");
    try {
      setAnswer(await ask(question));
      await loadWorkspace();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "问答失败");
    } finally {
      setLoading(false);
    }
  };

  const onUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setMessage("");
    try {
      const result = await uploadDocument(file);
      setMessage(`入库完成：新增 ${result.indexed_files} 个文件，跳过 ${result.skipped_files} 个重复文件，当前 ${result.total_chunks} 个切片。`);
      await loadWorkspace();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "上传失败");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  const openFile = async (fileId: string) => {
    setMessage("");
    try {
      setSelectedFile(await fetchFileDetail(fileId));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "文件详情读取失败");
    }
  };

  const removeFile = async () => {
    if (!selectedFile) return;
    const confirmed = window.confirm(`确认删除「${selectedFile.filename}」？删除后会同步移除原文件、索引切片和问答缓存。`);
    if (!confirmed) return;
    setDeleting(true);
    setMessage("");
    try {
      const result = await deleteKnowledgeFile(selectedFile.id);
      setSelectedFile(null);
      setAnswer(null);
      setMessage(`已删除：${result.deleted_file}，相关索引和缓存已清理。`);
      await loadWorkspace();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "删除失败");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Brain size={24} /></div>
          <div>
            <strong>MultiModal RAG</strong>
            <span>个人知识库问答系统</span>
          </div>
        </div>
        <nav>
          <a className="active" href="#upload"><UploadCloud size={18} />文档上传</a>
          <a href="#kb"><Database size={18} />知识库状态</a>
          <a href="#ask"><MessageSquareText size={18} />在线问答</a>
          <a href="#sources"><FileSearch size={18} />引用与评分</a>
        </nav>
        <div className="stack-card">
          <span>技术栈</span>
          <p>Python · LangChain · Chroma · BM25 · RAGAS · OCR · VLM</p>
        </div>
      </aside>

      <main className="main">
        <header>
          <div>
            <p className="eyebrow">2026.01 - 2026.04 · AI 应用开发</p>
            <h1>面向个人的多模态 RAG 知识库问答系统</h1>
          </div>
          <button className="icon-btn" onClick={loadWorkspace} title="刷新状态">
            <RefreshCw size={18} />
          </button>
        </header>

        {message && <div className="notice">{message}</div>}

        <section id="kb" className="metric-grid">
          <Metric icon={<FileText size={20} />} label="已入库文件" value={String(stats?.files ?? 0)} />
          <Metric icon={<Database size={20} />} label="知识切片" value={String(stats?.chunks ?? 0)} />
          <Metric icon={<Image size={20} />} label="模态类型" value={(stats?.modalities || []).join(" / ") || "-"} />
          <Metric icon={<Gauge size={20} />} label="缓存条目" value={String(stats?.cache_files ?? 0)} />
        </section>

        <section className="grid">
          <div className="stacked">
            <div id="upload" className="panel">
              <div className="panel-title">
                <div>
                  <p className="eyebrow">Offline Index</p>
                  <h2>上传 PDF / Markdown / 图片</h2>
                </div>
                <UploadCloud size={22} />
              </div>
              <label className="upload-box">
                {uploading ? <Loader2 className="spin" size={26} /> : <UploadCloud size={28} />}
                <strong>{uploading ? "正在解析、清洗并入库" : "选择文件上传"}</strong>
                <span>支持 PDF、Markdown、TXT、PNG、JPG；图片会进行 OCR 文本提取并生成 VLM 语义摘要。</span>
                <input type="file" accept=".pdf,.md,.markdown,.txt,.png,.jpg,.jpeg,.bmp,.webp" onChange={onUpload} />
              </label>
              <div className="flow-list">
                <span>文档解析</span>
                <span>清洗去重</span>
                <span>Hash 增量索引</span>
                <span>Chroma + BM25</span>
              </div>
            </div>

            <div className="panel file-panel">
              <div className="panel-title">
                <div>
                  <p className="eyebrow">Knowledge Files</p>
                  <h2>入库文件管理</h2>
                </div>
                <FileText size={22} />
              </div>
              <div className="file-list">
                {files.length ? files.map((file) => (
                  <button
                    className={selectedFile?.id === file.id ? "file-item active" : "file-item"}
                    key={file.id}
                    onClick={() => openFile(file.id)}
                  >
                    <span>{file.filename}</span>
                    <small>{file.modalities.join(" / ") || "-"} · {file.chunks_count} 切片</small>
                  </button>
                )) : <p className="empty-state">暂无入库文件</p>}
              </div>
            </div>
          </div>

          <div id="ask" className="panel ask-panel">
            <div className="panel-title">
              <div>
                <p className="eyebrow">Online QA</p>
                <h2>在线问答检索</h2>
              </div>
              <Search size={22} />
            </div>
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={5} />
            <button className="primary" onClick={submitQuestion} disabled={loading}>
              {loading ? <Loader2 className="spin" size={18} /> : <MessageSquareText size={18} />}
              生成答案
            </button>
          </div>
        </section>

        {selectedFile && (
          <section className="panel detail-panel">
            <div className="panel-title">
              <div>
                <p className="eyebrow">File Detail</p>
                <h2>{selectedFile.filename}</h2>
              </div>
              <button className="danger" onClick={removeFile} disabled={deleting}>
                {deleting ? <Loader2 className="spin" size={16} /> : <Trash2 size={16} />}
                删除文件
              </button>
            </div>
            <div className="file-meta">
              <span>类型：{selectedFile.modalities.join(" / ") || "-"}</span>
              <span>切片：{selectedFile.chunks_count}</span>
              <span>Hash：{selectedFile.hash.slice(0, 12)}</span>
              <span>{selectedFile.exists ? "原文件存在" : "原文件缺失"}</span>
            </div>
            <div className="chunk-list">
              {selectedFile.chunks.map((chunk, index) => (
                <article key={chunk.id}>
                  <strong>切片 {index + 1} · {chunk.modality}</strong>
                  <p>{chunk.preview}</p>
                </article>
              ))}
            </div>
          </section>
        )}

        <section className="grid lower">
          <div className="panel answer-panel">
            <div className="panel-title">
              <div>
                <p className="eyebrow">Generated Answer</p>
                <h2>答案</h2>
              </div>
              <Activity size={22} />
            </div>
            <pre>{answer?.answer || "上传资料并提问后，这里会显示基于检索上下文生成的答案。"}</pre>
            {answer && (
              <small>
                耗时 {answer.latency_ms}ms · {answer.cached ? "命中本地缓存" : "新生成"} ·
                答案来源 {answer.answer_provider || "local-template"}
                {answer.generation_error ? " · LLM 调用失败，已使用本地兜底" : ""}
              </small>
            )}
          </div>

          <div id="sources" className="panel">
            <div className="panel-title">
              <div>
                <p className="eyebrow">RAGAS-style Evaluation</p>
                <h2>引用来源与评分</h2>
              </div>
              <FileSearch size={22} />
            </div>
            <div className="score-list">
              {metrics.map(([label, value]) => (
                <div className="score-row" key={label}>
                  <span>{label}</span>
                  <strong>{Math.round(Number(value) * 100)}%</strong>
                  <div><i style={{ width: `${Math.round(Number(value) * 100)}%` }} /></div>
                </div>
              ))}
            </div>
            <div className="source-list">
              {(answer?.sources || []).map((source) => (
                <article key={`${source.filename}-${source.score}`}>
                  <div>
                    <strong>{source.filename}</strong>
                    <span>{source.modality} · score {source.score}</span>
                  </div>
                  <p>{source.preview}</p>
                </article>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="metric">
      <div>{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
