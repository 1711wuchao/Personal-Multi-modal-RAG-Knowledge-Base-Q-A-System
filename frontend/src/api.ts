export type Stats = {
  files: number;
  chunks: number;
  modalities: string[];
  cache_files: number;
};

export type KnowledgeFile = {
  id: string;
  source: string;
  filename: string;
  hash: string;
  exists: boolean;
  chunks_count: number;
  modalities: string[];
};

export type KnowledgeFileDetail = KnowledgeFile & {
  chunks: Array<{
    id: string;
    modality: string;
    preview: string;
    metadata: Record<string, string>;
  }>;
};

export type AskResponse = {
  question: string;
  answer: string;
  answer_provider?: string;
  generation_error?: string | null;
  sources: Array<{
    source: string;
    filename: string;
    modality: string;
    score: number;
    bm25_score: number;
    vector_score: number;
    preview: string;
  }>;
  metrics: {
    answer_accuracy: number;
    retrieval_quality: number;
    context_hit_rate: number;
  };
  latency_ms: number;
  cached: boolean;
};

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || "请求失败");
  }
  return payload as T;
}

export function fetchStats() {
  return request<Stats>("/api/stats");
}

export function fetchFiles() {
  return request<{ files: KnowledgeFile[] }>("/api/files");
}

export function fetchFileDetail(fileId: string) {
  return request<KnowledgeFileDetail>(`/api/files/${fileId}`);
}

export function deleteKnowledgeFile(fileId: string) {
  return request<{ status: string; deleted_file: string; id: string }>(`/api/files/${fileId}`, {
    method: "DELETE"
  });
}

export function ask(question: string) {
  return request<AskResponse>("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
}

export function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  return request<{ indexed_files: number; skipped_files: number; indexed_chunks: number; total_chunks: number }>(
    "/api/upload",
    { method: "POST", body: form }
  );
}
