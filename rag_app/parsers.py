from __future__ import annotations

from pathlib import Path
import markdown
import re

from PIL import Image
from pypdf import PdfReader
import pytesseract

from .models import ParsedDocument
from .text_utils import normalize_text


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TEXT_EXTS = {".md", ".markdown", ".txt"}


class DocumentParser:
    def parse_path(self, path: Path) -> list[ParsedDocument]:
        if path.is_dir():
            docs: list[ParsedDocument] = []
            for child in sorted(path.rglob("*")):
                if child.is_file() and self.supports(child):
                    docs.extend(self.parse_file(child))
            return docs
        return self.parse_file(path)

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in TEXT_EXTS | IMAGE_EXTS | {".pdf"}

    def parse_file(self, path: Path) -> list[ParsedDocument]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return [self._parse_pdf(path)]
        if suffix in TEXT_EXTS:
            return [self._parse_text(path)]
        if suffix in IMAGE_EXTS:
            return [self._parse_image(path)]
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        reader = PdfReader(str(path))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[page {idx}]\n{text}")
        return ParsedDocument(
            source_path=path,
            content=normalize_text("\n\n".join(pages)),
            modality="pdf",
            metadata={"pages": str(len(reader.pages))},
        )

    def _parse_text(self, path: Path) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8-sig", errors="ignore")
        if path.suffix.lower() in {".md", ".markdown"}:
            html = markdown.markdown(raw)
            raw = re.sub(r"<[^>]+>", " ", html)
        return ParsedDocument(source_path=path, content=normalize_text(raw), modality="markdown", metadata={})

    def _parse_image(self, path: Path) -> ParsedDocument:
        image = Image.open(path)
        try:
            ocr_text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        except Exception:
            ocr_text = ""
        vlm_summary = self._mock_vlm_summary(path, image, ocr_text)
        content = normalize_text(f"OCR文本:\n{ocr_text}\n\nVLM图片语义摘要:\n{vlm_summary}")
        return ParsedDocument(
            source_path=path,
            content=content,
            modality="image",
            metadata={"width": str(image.width), "height": str(image.height)},
        )

    def _mock_vlm_summary(self, path: Path, image: Image.Image, ocr_text: str) -> str:
        if ocr_text.strip():
            return f"图片 {path.name} 包含可识别文字，主题与文本内容相关。"
        return f"图片 {path.name} 尺寸为 {image.width}x{image.height}，可接入 Qwen-VL 等模型生成更细语义描述。"
