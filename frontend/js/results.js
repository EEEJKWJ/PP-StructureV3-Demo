import { saveCache, state } from "./state.js";
import { sanitizeTableHtml } from "./table-html.js";
import { LABEL_COLORS } from "./theme.js";

const cellsViewMode = {};
const tableHtmlMode = {};

export function escHtml(value, lineBreaks) {
  if (!value) return "";
  value = value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
  if (lineBreaks) value = value.replace(/\n/g, "<br>");
  return value;
}

export function switchResultView(mode) {
  state.resultView = mode;
  document
    .querySelectorAll(".view-switch span")
    .forEach((element) => element.classList.remove("active"));
  document
    .querySelector(`.view-switch span[onclick*="${mode}"]`)
    .classList.add("active");
  document.getElementById("resultList").style.display =
    mode === "visual" ? "block" : "none";
  document.getElementById("jsonView").style.display =
    mode === "json" ? "block" : "none";
  document.getElementById("jsonAllView").style.display =
    mode === "json-all" ? "block" : "none";
  if (mode === "json") renderJson();
  if (mode === "json-all") renderJsonAll();
  saveCache();
}

export function renderJson() {
  const view = document.getElementById("jsonView");
  const data = state.detectedPages[state.activePage];
  if (!data) {
    view.innerHTML = '<div class="loading">暂无数据</div>';
    return;
  }
  const structured = data.structured || data;
  view.innerHTML =
    `<pre>${escHtml(JSON.stringify(structured, null, 2))}</pre>`;
}

export function renderJsonAll() {
  const view = document.getElementById("jsonAllView");
  if (!state.allPagesData) {
    view.innerHTML =
      '<div class="loading">请先执行「全部检测」</div>';
    return;
  }
  const allStructured = state.allPagesData.pages.map(
    (page) => page.structured,
  );
  const merges = state.allPagesData.table_merges;
  const output = { pages: allStructured };
  if (merges && merges.length > 0) {
    output.table_merges = merges;
  }
  view.innerHTML =
    `<pre>${escHtml(JSON.stringify(output, null, 2))}</pre>`;
}

export function renderResults() {
  const list = document.getElementById("resultList");
  const data = state.detectedPages[state.activePage];

  if (!data) {
    list.innerHTML =
      '<div class="loading">点击「检测当前页」开始</div>';
    return;
  }

  const sourceTag = document.getElementById("sourceTag");
  if (data.text_source === "ocr") {
    sourceTag.innerHTML =
      '<span class="source-tag ocr">OCR 识别</span>';
  } else if (data.text_source === "pdf") {
    sourceTag.innerHTML =
      '<span class="source-tag pdf">PDF 文字层</span>';
  } else {
    sourceTag.innerHTML = "";
  }

  if (data.blocks.length === 0) {
    list.innerHTML =
      '<div class="loading">未检测到任何区域</div>';
    return;
  }

  list.innerHTML = data.blocks
    .map((block, index) => {
      const color = LABEL_COLORS[block.label] || "#78909c";
      const [x1, y1, x2, y2] = block.bbox.map((value) =>
        Math.round(value),
      );
      let extra = "";

      if (block.label === "table") {
        if (block._merged_into) {
          extra =
            `<div class="merged-notice">此表格已合并至第 ${
              block._merged_into[0] !== undefined ? "前" : ""
            }页的表格中</div>`;
        } else if (block.cells && block.cells.length > 0) {
          extra = renderCellsTable(block.cells, index);
          if (
            block.merged_from_pages
            && block.merged_from_pages.length > 1
          ) {
            extra +=
              `<div class="merged-notice">跨页合并表格（来自第 ${
                block.merged_from_pages.join("、")
              } 页）</div>`;
          }
        } else if (block.cells === null) {
          extra =
            '<div class="image-hint">扫描件表格：已通过 OCR 识别文字，但无法解析单元格行列结构</div>';
        }
        if (block.table_html) {
          extra += renderTableHtml(block.table_html, index);
        }
        if (block.table_model || block.table_elapsed_ms !== undefined) {
          extra += renderTableMetadata(block);
        }
      }

      if (block.toc_items && block.toc_items.length > 0) {
        extra = renderTocBlock(block.toc_items, index);
      }

      if (!extra && block.ocr_text) {
        extra =
          `<div class="block-text">${escHtml(block.ocr_text, true)}</div>`;
      }

      return `<div class="block-item" data-block="${index}" style="border-left-color:${color}; background:${color}10" onclick="highlightBlock(${index})">
      <span class="order-num">#${block.order}</span>
      <div class="label" style="color:${color}">${block.label}</div>
      <div class="bbox-info">[${x1}, ${y1}, ${x2}, ${y2}] — ${x2 - x1}×${y2 - y1}px</div>
      ${extra}
    </div>`;
    })
    .join("");
}

function renderTableMetadata(block) {
  const model = block.table_model || {};
  const modelName = model.name || "PDF 文字层";
  const backend = model.backend
    ? ` · ${model.backend}`
    : "";
  const device = model.device
    ? ` · ${model.device}`
    : "";
  const elapsed = Number.isFinite(block.table_elapsed_ms)
    ? ` · ${block.table_elapsed_ms.toFixed(2)} ms`
    : "";
  const shape =
    block.table_row_count && block.table_col_count
      ? ` · ${block.table_row_count}×${block.table_col_count}`
      : "";
  return `<div class="table-metadata">${escHtml(
    `${modelName}${backend}${device}${elapsed}${shape}`,
  )}</div>`;
}

export function renderTableHtml(tableHtml, blockIndex) {
  const id = `table-html-${blockIndex}`;
  const mode = tableHtmlMode[blockIndex] || "render";
  let body = "";

  if (mode === "source") {
    body =
      `<pre class="table-html-source">${escHtml(tableHtml)}</pre>`;
  } else {
    const sanitized = sanitizeTableHtml(tableHtml);
    body = sanitized.error
      ? `<div class="table-html-error">${escHtml(sanitized.error)}</div>`
      : `<div class="table-html-render">${sanitized.html}</div>`;
  }

  let html =
    `<div class="table-html-section"><div class="cells-toggle" onclick="event.stopPropagation(); toggleCells('${id}')">`;
  html += "▶ 表格 HTML（点击展开）</div>";
  html += `<div id="${id}" style="display:none">`;
  html += '<div class="html-mode-switch" onclick="event.stopPropagation()">';
  html +=
    `<span class="${mode === "render" ? "active" : ""}" onclick="switchTableHtmlMode(${blockIndex}, 'render')">渲染</span>`;
  html +=
    `<span class="${mode === "source" ? "active" : ""}" onclick="switchTableHtmlMode(${blockIndex}, 'source')">源码</span>`;
  html += "</div>";
  html += body;
  html += "</div></div>";
  return html;
}

export function switchTableHtmlMode(blockIndex, mode) {
  tableHtmlMode[blockIndex] = mode;
  renderResults();
  const element = document.getElementById(
    `table-html-${blockIndex}`,
  );
  if (element) {
    element.style.display = "block";
    const toggle = element.previousElementSibling;
    toggle.innerHTML = toggle.innerHTML
      .replace("▶", "▼")
      .replace("点击展开", "点击收起");
  }
}

export function renderCellsTable(cells, blockIndex) {
  const id = `cells-${blockIndex}`;
  const mode = cellsViewMode[blockIndex] || "grid";

  let html =
    `<div class="cells-toggle" onclick="event.stopPropagation(); toggleCells('${id}')">`;
  html += `▶ ${cells.length} 个单元格（点击展开）`;
  html +=
    '<span class="cells-mode-switch" onclick="event.stopPropagation()">';
  html +=
    `<span class="${mode === "grid" ? "active" : ""}" onclick="switchCellsMode(${blockIndex}, 'grid')">表格</span>`;
  html +=
    `<span class="${mode === "tree" ? "active" : ""}" onclick="switchCellsMode(${blockIndex}, 'tree')">树形</span>`;
  html += "</span></div>";

  html += `<div id="${id}" style="display:none">`;
  html += mode === "grid"
    ? renderGrid(cells, blockIndex)
    : renderTree(cells, blockIndex);
  html += "</div>";
  return html;
}

export function switchCellsMode(blockIndex, mode) {
  cellsViewMode[blockIndex] = mode;
  renderResults();
  const id = `cells-${blockIndex}`;
  const element = document.getElementById(id);
  if (element) {
    element.style.display = "block";
    const toggle = element.previousElementSibling;
    toggle.innerHTML = toggle.innerHTML
      .replace("▶", "▼")
      .replace("点击展开", "点击收起");
  }
}

export function renderGrid(cells, blockIndex) {
  const maxRow = Math.max(...cells.map((cell) => cell.row));
  const maxCol = Math.max(...cells.map((cell) => cell.col));
  const grid = Array.from(
    { length: maxRow + 1 },
    () => Array(maxCol + 1).fill(""),
  );
  for (const cell of cells) {
    grid[cell.row][cell.col] = cell.text;
  }

  let html = '<table class="cells-table">';
  for (let row = 0; row <= maxRow; row++) {
    html += "<tr>";
    for (let col = 0; col <= maxCol; col++) {
      const text = grid[row][col];
      html +=
        `<td data-block="${blockIndex}" data-row="${row}" data-col="${col}" title="${escHtml(text)}" onclick="event.stopPropagation(); highlightCell(${blockIndex}, ${row}, ${col})">${escHtml(text, true)}</td>`;
    }
    html += "</tr>";
  }
  html += "</table>";
  return html;
}

export function renderTree(cells, blockIndex) {
  const maxRow = Math.max(...cells.map((cell) => cell.row));
  const maxCol = Math.max(...cells.map((cell) => cell.col));
  const grid = Array.from(
    { length: maxRow + 1 },
    () => Array(maxCol + 1).fill(null),
  );
  for (const cell of cells) {
    grid[cell.row][cell.col] = cell;
  }

  const colCount = maxCol + 1;
  const lastCol = maxCol;

  function getText(row, col) {
    for (let currentRow = row; currentRow >= 0; currentRow--) {
      const cell = grid[currentRow] && grid[currentRow][col];
      if (cell && cell.text) return cell;
    }
    return null;
  }

  const roots = [];
  const levelMap = [];

  for (let row = 1; row <= maxRow; row++) {
    const rowCells = [];
    for (let col = 0; col < colCount; col++) {
      rowCells.push(getText(row, col));
    }

    let changedLevel = -1;
    for (let col = 0; col < colCount - 1; col++) {
      const cell = grid[row] && grid[row][col];
      if (cell && cell.text && changedLevel === -1) {
        changedLevel = col;
      }
    }
    if (changedLevel === -1) changedLevel = colCount - 2;

    for (let col = changedLevel; col < colCount - 1; col++) {
      const cellData = rowCells[col];
      if (!cellData || !cellData.text) continue;

      const node = {
        text: cellData.text,
        row: cellData.row,
        col: cellData.col,
        children: [],
        tag: null,
      };

      if (col === 0) {
        roots.push(node);
      } else {
        const parent = levelMap[col - 1];
        if (parent) parent.children.push(node);
        else roots.push(node);
      }
      levelMap[col] = node;
      for (let childCol = col + 1; childCol < colCount; childCol++) {
        levelMap[childCol] = null;
      }
    }

    const tagCell = rowCells[lastCol];
    const deepest =
      levelMap[colCount - 2]
      || levelMap[colCount - 3]
      || roots[roots.length - 1];
    if (deepest && tagCell && tagCell.text) {
      deepest.tag = tagCell.text;
      deepest.tagRow = tagCell.row;
      deepest.tagCol = tagCell.col;
    }
  }

  function renderNode(node) {
    const hasChildren = node.children.length > 0;
    let html = '<div class="tree-node">';

    if (hasChildren) {
      html +=
        '<span class="tree-toggle" onclick="event.stopPropagation(); toggleTreeNode(this)">▼</span>';
      html +=
        `<span class="tree-label" data-block="${blockIndex}" data-row="${node.row}" data-col="${node.col}" onclick="event.stopPropagation(); highlightCell(${blockIndex}, ${node.row}, ${node.col})">${escHtml(node.text, true)}</span>`;
      if (node.tag) {
        html +=
          ` <span class="tree-tag">${escHtml(node.tag)}</span>`;
      }
      html += '<div class="tree-children">';
      for (const child of node.children) {
        html += renderNode(child);
      }
      html += "</div>";
    } else {
      const lines = node.text
        .split("\n")
        .filter((line) => line.trim());
      const tags = node.tag
        ? node.tag.split("\n").filter((line) => line.trim())
        : [];

      if (lines.length <= 1) {
        html +=
          `<div class="tree-leaf" data-block="${blockIndex}" data-row="${node.row}" data-col="${node.col}" onclick="event.stopPropagation(); highlightCell(${blockIndex}, ${node.row}, ${node.col})">`;
        html += `<span>${escHtml(node.text)}</span>`;
        if (node.tag) {
          html +=
            `<span class="tree-tag">${escHtml(node.tag)}</span>`;
        }
        html += "</div>";
      } else {
        for (let index = 0; index < lines.length; index++) {
          const lineTag = tags.length === 1
            ? tags[0]
            : tags[index] || "";
          html +=
            `<div class="tree-leaf" data-block="${blockIndex}" data-row="${node.row}" data-col="${node.col}" onclick="event.stopPropagation(); highlightCell(${blockIndex}, ${node.row}, ${node.col})">`;
          html += `<span>${escHtml(lines[index])}</span>`;
          if (lineTag) {
            html +=
              `<span class="tree-tag">${escHtml(lineTag)}</span>`;
          }
          html += "</div>";
        }
      }
    }
    html += "</div>";
    return html;
  }

  let html = '<div class="cells-tree">';
  for (const root of roots) {
    html += renderNode(root);
  }
  html += "</div>";
  return html;
}

export function renderTocBlock(items, blockIndex) {
  const id = `toc-${blockIndex}`;
  let html =
    `<div class="cells-toggle" onclick="event.stopPropagation(); toggleCells('${id}')">`;
  html += `▶ ${items.length} 条目录（点击展开）</div>`;
  html += `<div id="${id}" style="display:none"><div class="toc-list">`;
  items.forEach((item, index) => {
    const page = item.page !== null
      ? `<span>${item.page}</span>`
      : "";
    html +=
      `<div class="toc-item" data-block="${blockIndex}" data-toc="${index}" onclick="event.stopPropagation(); highlightTocItem(${blockIndex}, ${index})">`;
    html +=
      `<span class="toc-title">${escHtml(item.title)}</span>`;
    html += `<span class="toc-page">${page}</span>`;
    html += "</div>";
  });
  html += "</div></div>";
  return html;
}

export function toggleCells(id) {
  const element = document.getElementById(id);
  if (!element) return;
  const toggle = element.previousElementSibling;
  if (element.style.display === "none") {
    element.style.display = "block";
    toggle.innerHTML = toggle.innerHTML
      .replace("▶", "▼")
      .replace("点击展开", "点击收起");
  } else {
    element.style.display = "none";
    toggle.innerHTML = toggle.innerHTML
      .replace("▼", "▶")
      .replace("点击收起", "点击展开");
  }
}

export function toggleTreeNode(toggle) {
  const children =
    toggle.parentElement.querySelector(".tree-children");
  if (!children) return;
  if (children.style.display === "none") {
    children.style.display = "block";
    toggle.textContent = "▼";
  } else {
    children.style.display = "none";
    toggle.textContent = "▶";
  }
}
