"""Model-specific document OCR prompts and output adapters."""

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable
import re


class TableParser(HTMLParser):
    """Extract data cells only: model HTML is never sent to an HTML execution sink."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.blocked = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "iframe", "object", "svg", "math"):
            self.blocked += 1
        if self.blocked:
            return
        if tag == "tr":
            self.row = []
            self.rows.append(self.row)
        elif tag in ("td", "th"):
            if self.row is None:
                return
            attrs = dict(attrs)
            self.cell = dict(text="", header=tag == "th")
            for span in ("colspan", "rowspan"):
                value = attrs.get(span) or ""
                self.cell[span] = (
                    min(int(value), 100) if value.isascii() and value.isdigit() and 0 < len(value) < 4 else 1
                )
                self.cell[span] = max(1, self.cell[span])
            self.row.append(self.cell)
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += "\n"

    def handle_endtag(self, tag):
        if tag in ("script", "style", "iframe", "object", "svg", "math") and self.blocked:
            self.blocked -= 1
        elif not self.blocked:
            if tag in ("td", "th"):
                self.cell = None
            elif tag == "tr":
                self.row = None

    def handle_data(self, data):
        if not self.blocked and self.cell is not None:
            self.cell["text"] += data


def extract_tables(text):
    tables = []
    for match in re.finditer(r"<table\b[^>]*>[\s\S]*?</table\s*>", text, re.I):
        parser = TableParser()
        parser.feed(match.group())
        tables.append(dict(source=match.group(), rows=parser.rows))
    return tables


def parse_hunyuan(text):
    # Keep the exact generated stream in `raw`; normalize only the document view.
    for number, tag in ((110, "quad"), (111, "/quad"), (112, "ref"), (113, "/ref"), (114, "pFig"), (115, "/pFig")):
        text = text.replace(f"<｜hy_place▁holder▁no▁{number}｜>", f"<{tag}>")
    text = re.sub(r"<quad>.*?</quad>", "", text, flags=re.S)
    text = re.sub(r"<quad>.*\Z", "", text, flags=re.S)
    text = re.sub(r"</?(?:ref|pFig)>", "", text)
    text = re.sub(r"<｜[^｜]*｜>", "", text)
    text = re.sub(r"<｜.*\Z", "", text, flags=re.S)
    return dict(markdown=text.strip(), boxes=[])


def parse_deepseek(text):
    boxes = []
    pattern = r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>"
    for label, coordinates in re.findall(pattern, text, re.S):
        for values in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", coordinates):
            x1, y1, x2, y2 = map(int, values)
            if 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000:
                boxes.append(dict(label=label, x=x1 / 10, y=y1 / 10, width=(x2 - x1) / 10, height=(y2 - y1) / 10))
    markdown = re.sub(pattern, "", text, flags=re.S)
    # Cancellation can end inside a grounding marker. Keep it only in raw output.
    markdown = re.sub(r"<\|ref\|>.*\Z", "", markdown, flags=re.S)
    markdown = re.sub(r"<\|(?:endoftext|eos)\|>", "", markdown)
    return dict(markdown=markdown.strip(), boxes=boxes)


@dataclass(frozen=True)
class OCRAdapter:
    name: str
    prompt: str
    layout: bool
    parse: Callable[[str], dict]


DEEPSEEK = OCRAdapter("deepseek", "{image}\n<|grounding|>Convert the document to markdown.", True, parse_deepseek)
HUNYUAN = OCRAdapter(
    "hunyuan",
    "<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>{image}"
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。<｜hy_User｜>",
    False,
    parse_hunyuan,
)


def ocr_adapter(model):
    name = str(getattr(model, "model_path", "")) + " " + str(getattr(model, "model_id", ""))
    if re.search(r"deepseek[-_]?ocr", name, re.I):
        return DEEPSEEK
    if re.search(r"hunyuan[-_]?ocr", name, re.I):
        return HUNYUAN
    return None
