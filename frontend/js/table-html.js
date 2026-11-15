const ALLOWED_TAGS = new Set([
  "table",
  "thead",
  "tbody",
  "tfoot",
  "tr",
  "th",
  "td",
  "caption",
  "colgroup",
  "col",
  "br",
]);

const ALLOWED_ATTRIBUTES = new Set([
  "rowspan",
  "colspan",
  "scope",
  "title",
  "abbr",
  "headers",
]);

const BLOCKED_SUBTREES = new Set([
  "script",
  "style",
  "iframe",
  "object",
  "embed",
  "svg",
  "math",
]);

function boundedSpan(value) {
  if (value === null || value === "") return null;
  const number = Number.parseInt(value, 10);
  if (!Number.isFinite(number) || number < 1) return null;
  return String(Math.min(number, 1000));
}

function cloneSafeNode(node, outputDocument) {
  if (node.nodeType === Node.TEXT_NODE) {
    return outputDocument.createTextNode(node.textContent || "");
  }
  if (node.nodeType !== Node.ELEMENT_NODE) {
    return null;
  }

  const tag = node.tagName.toLowerCase();
  if (BLOCKED_SUBTREES.has(tag)) {
    return null;
  }

  const fragment = outputDocument.createDocumentFragment();
  if (!ALLOWED_TAGS.has(tag)) {
    for (const child of node.childNodes) {
      const safeChild = cloneSafeNode(child, outputDocument);
      if (safeChild) fragment.appendChild(safeChild);
    }
    return fragment;
  }

  const element = outputDocument.createElement(tag);
  for (const attribute of node.attributes) {
    const name = attribute.name.toLowerCase();
    if (!ALLOWED_ATTRIBUTES.has(name)) continue;
    if (name === "rowspan" || name === "colspan") {
      const value = boundedSpan(attribute.value);
      if (value) element.setAttribute(name, value);
      continue;
    }
    element.setAttribute(name, attribute.value);
  }
  for (const child of node.childNodes) {
    const safeChild = cloneSafeNode(child, outputDocument);
    if (safeChild) element.appendChild(safeChild);
  }
  return element;
}

export function sanitizeTableHtml(tableHtml) {
  if (!tableHtml || typeof tableHtml !== "string") {
    return { html: "", error: "没有可用的表格 HTML" };
  }

  try {
    const parsed = new DOMParser().parseFromString(
      tableHtml,
      "text/html",
    );
    const sourceTable = parsed.querySelector("table");
    if (!sourceTable) {
      return { html: "", error: "HTML 中没有有效的 table" };
    }

    const outputDocument = document.implementation.createHTMLDocument("");
    const safeTable = cloneSafeNode(sourceTable, outputDocument);
    if (!safeTable || safeTable.nodeName.toLowerCase() !== "table") {
      return { html: "", error: "表格 HTML 清理失败" };
    }
    return { html: safeTable.outerHTML, error: "" };
  } catch {
    return { html: "", error: "表格 HTML 解析失败" };
  }
}
