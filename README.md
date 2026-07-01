# 面向个人的多模态 RAG 知识库问答系统

时间：2026.01 - 2026.04  
角色：独立开发  
技术栈：Python、LangChain、Chroma、BM25、RAGAS、OCR、VLM、FastAPI、React、TypeScript

## 项目功能

- 支持导入 PDF / Markdown / 图片
- PDF、Markdown 文本解析
- 图片 OCR 文本提取，模拟 VLM 图片语义摘要接口
- 文档清洗、去重、元数据绑定
- 文档 Hash 增量索引，避免重复入库
- BM25 关键词检索 + Chroma 向量检索
- 多路召回 + 简单重排
- 在线问答，根据检索内容调用 LangChain 大模型生成答案
- 简易评测模块，模拟 RAGAS 的准确率、召回质量、上下文命中率
- 本地缓存，提升重复问答速度，并在知识库更新后自动失效
- 前端界面：上传文档、查看知识库、提问、查看引用来源和评分

## 本地配置

项目会读取根目录 `.env`，用于配置大模型 API：

```env
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

如果没有配置密钥，系统会自动使用本地模板答案兜底，检索、引用来源和评分仍可正常演示。

## 启动

后端：

```powershell
cd C:\Users\Administrator\Documents\agent\personal-multimodal-rag
.\.venv\Scripts\python.exe -m uvicorn rag_app.api:app --host 127.0.0.1 --port 8010
```

前端：

```powershell
cd C:\Users\Administrator\Documents\agent\personal-multimodal-rag\frontend
npm run dev -- --port 5174
```

打开：

```text
http://127.0.0.1:5174
```

## 命令行演示

```powershell
.\.venv\Scripts\python.exe -m rag_app.cli ingest demo_docs
.\.venv\Scripts\python.exe -m rag_app.cli ask "个人知识库支持什么？"
```

## 测试

后端：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

前端：

```powershell
cd frontend
npm run build
```

## 简历描述

聚焦个人知识管理场景，基于 Python、LangChain 搭建轻量化多模态 RAG 知识库问答系统，支持 PDF、Markdown、图片等常见格式文件解析入库。围绕离线索引、在线问答检索、效果测评、增量更新等核心环节完成全链路优化，解决传统 RAG 问答不准、响应慢、无法增量更新等问题，实现可落地的个人知识库工具。
