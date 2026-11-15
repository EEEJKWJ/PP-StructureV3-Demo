import {
  detectAllPages,
  detectPage,
  getModelStatus,
  getTextSource,
  uploadDocument,
} from "./api.js";
import {
  clearHighlights,
  highlightBlock,
  highlightCell,
  highlightScanCell,
  highlightTocItem,
} from "./interactions.js";
import { drawBoxes } from "./overlay.js";
import {
  renderJson,
  renderJsonAll,
  renderResults,
  switchCellsMode,
  switchResultView,
  toggleCells,
  toggleTreeNode,
} from "./results.js";
import {
  loadCache,
  runtime,
  saveCache,
  state,
} from "./state.js";

Object.assign(window, {
  switchResultView,
  selectPage,
  onImageLoaded,
  highlightBlock,
  highlightCell,
  highlightScanCell,
  highlightTocItem,
  toggleCells,
  switchCellsMode,
  toggleTreeNode,
});

function renderSidebar() {
  const sidebar = document.getElementById("sidebar");
  sidebar.innerHTML = state.pages
    .map((_, index) => {
      const pageNumber = index + 1;
      const detected = state.detectedPages[pageNumber]
        ? " detected"
        : "";
      const active =
        state.activePage === pageNumber ? " active" : "";
      return `<div class="page-thumb${active}${detected}" onclick="selectPage(${pageNumber})">第 ${pageNumber} 页</div>`;
    })
    .join("");
}

function selectPage(pageNumber) {
  state.activePage = pageNumber;
  const preview = document.getElementById("preview");
  const filename = state.pages[pageNumber - 1];

  preview.innerHTML = `<div class="img-container" id="imgContainer">
    <img id="pageImg" src="/api/images/${state.docId}/${filename}" onload="onImageLoaded()">
  </div>`;

  document.getElementById("sourceTag").innerHTML = "";
  document.getElementById("detectBtn").disabled = !runtime.modelReady;
  renderSidebar();
  renderResults();
  if (state.resultView === "json") renderJson();
  switchResultView(state.resultView);
  saveCache();
}

function onImageLoaded() {
  if (state.detectedPages[state.activePage]) {
    drawBoxes(state.detectedPages[state.activePage]);
  }
}

async function handleFile(file) {
  document.getElementById("sourceTag").innerHTML = "";
  document.getElementById("resultList").innerHTML =
    '<div class="loading">上传中...</div>';

  const data = await uploadDocument(file);
  state.docId = data.doc_id;
  state.pages = data.pages;
  state.isPdf = data.is_pdf;
  state.detectedPages = {};
  state.allPagesData = null;
  document.getElementById("jsonAllTab").style.display = "none";
  document.getElementById("detectAllProgress").style.display = "none";

  renderSidebar();
  selectPage(1);
  saveCache();
  if (runtime.modelReady) {
    document.getElementById("detectAllBtn").disabled = false;
  }
}

async function detectCurrent() {
  const button = document.getElementById("detectBtn");
  button.disabled = true;
  button.textContent = "检测中...";
  document.getElementById("resultList").innerHTML =
    '<div class="loading">正在做版面检测...</div>';

  try {
    const sourceData = await getTextSource(
      state.docId,
      state.activePage,
    );
    const sourceTag = document.getElementById("sourceTag");
    if (sourceData.text_source === "ocr") {
      sourceTag.innerHTML =
        '<span class="source-tag ocr">OCR 识别</span>';
      document.getElementById("resultList").innerHTML =
        '<div class="loading">正在做版面检测 + OCR 识别（较慢）...</div>';
    } else {
      sourceTag.innerHTML =
        '<span class="source-tag pdf">PDF 文字层</span>';
    }

    const data = await detectPage(state.docId, state.activePage);
    state.detectedPages[state.activePage] = data;
    drawBoxes(data);
    renderResults();
    renderJson();
    renderSidebar();
    saveCache();
  } catch (error) {
    document.getElementById("resultList").innerHTML =
      `<div class="loading">检测失败: ${error.message}</div>`;
  }
  button.textContent = "检测当前页";
  button.disabled = false;
}

async function detectAll() {
  const button = document.getElementById("detectAllBtn");
  const detectButton = document.getElementById("detectBtn");
  button.disabled = true;
  detectButton.disabled = true;
  button.textContent = "检测中...";

  const progressElement =
    document.getElementById("detectAllProgress");
  const totalPages = state.pages.length;
  progressElement.style.display = "block";
  progressElement.innerHTML = `<div class="detect-all-progress">
    <div class="progress-text">正在检测所有页面... 0/${totalPages}</div>
    <div class="progress-bar"><div class="fill" style="width:0%"></div></div>
  </div>`;

  try {
    for (let page = 1; page <= totalPages; page++) {
      progressElement.querySelector(".progress-text").textContent =
        `正在检测第 ${page}/${totalPages} 页...`;
      progressElement.querySelector(".fill").style.width =
        `${((page - 0.5) / totalPages) * 100}%`;

      if (!state.detectedPages[page]) {
        state.detectedPages[page] = await detectPage(
          state.docId,
          page,
        );
      }
      progressElement.querySelector(".fill").style.width =
        `${(page / totalPages) * 100}%`;
    }

    progressElement.querySelector(".progress-text").textContent =
      "正在做跨页表格合并分析...";
    const allData = await detectAllPages(state.docId);
    state.allPagesData = allData;

    for (const pageData of allData.pages) {
      state.detectedPages[pageData.page] = pageData;
    }

    const merges = allData.table_merges;
    if (merges && merges.length > 0) {
      progressElement.innerHTML = `<div class="detect-all-progress">
        <div class="progress-text" style="color:#2e7d32">
          ✓ 全部 ${totalPages} 页检测完成，${merges.length} 处跨页表格已合并
        </div>
        <div style="font-size:11px;color:#666;margin-top:4px">${merges.map(
          (merge) =>
            `第${merge.from_page}页续表 → 第${merge.into_page}页（+${merge.rows_added}行${merge.header_skipped ? "，去重表头" : ""}）`,
        ).join("<br>")}</div>
      </div>`;
    } else {
      progressElement.innerHTML = `<div class="detect-all-progress">
        <div class="progress-text" style="color:#2e7d32">
          ✓ 全部 ${totalPages} 页检测完成，未发现跨页表格
        </div>
      </div>`;
    }

    document.getElementById("jsonAllTab").style.display = "";
    renderSidebar();
    if (state.activePage) {
      drawBoxes(state.detectedPages[state.activePage]);
      renderResults();
      renderJson();
    }
    saveCache();
  } catch (error) {
    progressElement.innerHTML = `<div class="detect-all-progress">
      <div class="progress-text" style="color:#ea4335">检测失败: ${error.message}</div>
    </div>`;
  }
  button.textContent = "全部检测";
  button.disabled = false;
  detectButton.disabled = false;
}

async function pollModelStatus() {
  const element = document.getElementById("modelStatus");
  while (true) {
    try {
      const data = await getModelStatus();
      if (data.status === "ready") {
        const ocrLabel = data.ocr_model
          ? ` | OCR: ${data.ocr_model}`
          : "";
        const gpuLabel = data.gpu
          ? ` | GPU: ${data.gpu_name}`
          : " | CPU";
        const tableLabel =
          data.table_engine && data.table_engine !== "none"
            ? ` | 表格: ${
              data.table_engine === "paddle"
                ? "PP-TableMagic V2"
                : "RapidTable"
            }`
            : "";
        element.textContent =
          "模型就绪" + gpuLabel + ocrLabel + tableLabel;
        element.style.background = "#34a853";
        if (!runtime.modelReady) {
          runtime.modelReady = true;
          document.getElementById("detectBtn").disabled =
            !state.activePage;
          document.getElementById("detectAllBtn").disabled =
            !state.docId;
        }
        if (
          data.table_engine
          && data.table_engine !== "none"
        ) {
          return;
        }
      } else if (data.status === "error") {
        element.textContent = "模型加载失败";
        element.style.background = "#ea4335";
        return;
      }
    } catch {
      // 保持轮询，模型状态接口可能仍在启动。
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

function setupUpload() {
  const fileInput = document.getElementById("fileInput");
  const uploadBox = document.getElementById("uploadBox");

  uploadBox.addEventListener("click", () => fileInput.click());
  uploadBox.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadBox.style.borderColor = "#1a73e8";
  });
  uploadBox.addEventListener("dragleave", () => {
    uploadBox.style.borderColor = "#ccc";
  });
  uploadBox.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadBox.style.borderColor = "#ccc";
    handleFile(event.dataTransfer.files[0]);
  });
  document
    .getElementById("headerFileInput")
    .addEventListener("change", function () {
      if (this.files[0]) handleFile(this.files[0]);
    });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
}

function setupResizePanel() {
  const panel = document.getElementById("resultPanel");
  const handle = document.getElementById("resizeHandle");
  let dragging = false;
  let startX;
  let startWidth;

  handle.addEventListener("mousedown", (event) => {
    dragging = true;
    startX = event.clientX;
    startWidth = panel.offsetWidth;
    handle.classList.add("active");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    event.preventDefault();
  });

  document.addEventListener("mousemove", (event) => {
    if (!dragging) return;
    const newWidth = startWidth - (event.clientX - startX);
    panel.style.flex = "none";
    panel.style.width =
      Math.max(200, Math.min(800, newWidth)) + "px";
  });

  document.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("active");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    if (state.activePage && state.detectedPages[state.activePage]) {
      drawBoxes(state.detectedPages[state.activePage]);
    }
  });
}

function setupEvents() {
  document
    .getElementById("detectBtn")
    .addEventListener("click", detectCurrent);
  document
    .getElementById("detectAllBtn")
    .addEventListener("click", detectAll);

  document.addEventListener("click", (event) => {
    if (
      !event.target.closest(
        ".bbox-overlay, .cell-overlay, .toc-overlay, .block-item, .cells-table, .toc-item",
      )
    ) {
      clearHighlights();
    }
  });

  window.addEventListener("resize", () => {
    if (state.activePage && state.detectedPages[state.activePage]) {
      drawBoxes(state.detectedPages[state.activePage]);
    }
  });
}

setupUpload();
setupResizePanel();
setupEvents();
pollModelStatus();

if (loadCache()) {
  renderSidebar();
  selectPage(state.activePage || 1);
  if (state.allPagesData) {
    document.getElementById("jsonAllTab").style.display = "";
  }
}

document.documentElement.dataset.frontendReady = "true";
