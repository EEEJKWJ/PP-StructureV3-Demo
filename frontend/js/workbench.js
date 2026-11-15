import {
  datasetSampleImageUrl,
  listDatasetSamples,
  listDatasets,
  runDatasetSample,
} from "./api.js";
import { escHtml } from "./results.js";
import { sanitizeTableHtml } from "./table-html.js";

const workbenchState = {
  initialized: false,
  loading: false,
  running: false,
  datasets: [],
  datasetId: null,
  offset: 0,
  limit: 20,
  page: null,
  selectedSampleId: null,
  runResult: null,
  predictionMode: "render",
  comparisonMode: "prediction",
  error: "",
};

function selectedSample() {
  return workbenchState.page?.samples?.find(
    (sample) => sample.sample_id === workbenchState.selectedSampleId,
  ) || null;
}

function selectedDataset() {
  return workbenchState.datasets.find(
    (dataset) => dataset.id === workbenchState.datasetId,
  ) || null;
}

function statusLabel(status) {
  return {
    success: "识别成功",
    no_table: "未检出表格",
    failed: "识别失败",
  }[status] || status || "未运行";
}

function renderMeta(sample, result) {
  if (!sample) return "";
  const prediction = result?.prediction || {};
  const model = prediction.model || {};
  const cellCount = prediction.cells?.length || 0;
  const shape =
    prediction.row_count && prediction.col_count
      ? ` · ${prediction.row_count}×${prediction.col_count}`
      : "";
  return `<div class="workbench-meta">
    ${escHtml(sample.file_name)}<br>
    #${sample.global_index} · ${sample.width}×${sample.height}px ·
    bbox [${(sample.table_bbox || []).map((value) => Math.round(value)).join(", ")}]<br>
    ${model.name
      ? `${escHtml(model.name)} · ${escHtml(model.backend || "")} · ${escHtml(model.device || "")}`
      : "尚未运行"}
    ${prediction.elapsed_ms !== undefined
      ? ` · ${Number(prediction.elapsed_ms).toFixed(2)} ms`
      : ""}
    ${result ? ` · ${cellCount} 个单元格${shape}` : ""}
  </div>`;
}

function renderPreview(sample, result) {
  if (!sample) {
    return '<div class="workbench-empty">请选择样本</div>';
  }
  const imageUrl = datasetSampleImageUrl(
    workbenchState.datasetId,
    sample.sample_id,
  );
  return `<div class="workbench-preview">
    <img src="${imageUrl}" alt="sample">
    ${renderMeta(sample, result)}
  </div>`;
}

function renderHtmlPayload(html, mode) {
  if (!html) {
    return '<div class="workbench-empty">没有 HTML 输出</div>';
  }
  if (mode === "source") {
    return `<pre class="table-html-source">${escHtml(html)}</pre>`;
  }
  const sanitized = sanitizeTableHtml(html);
  if (sanitized.error) {
    return `<div class="workbench-error">${escHtml(sanitized.error)}</div>`;
  }
  return `<div class="table-html-render">${sanitized.html}</div>`;
}

function renderResult(sample, result) {
  const groundTruth = result?.ground_truth || sample?.ground_truth || {};
  const hasTruthHtml = Boolean(groundTruth.html);
  const prediction = result?.prediction || {};
  const status = result?.status || "idle";
  const activePayload =
    workbenchState.comparisonMode === "truth"
      ? groundTruth
      : prediction;

  let content = "";
  if (workbenchState.running) {
    content = '<div class="workbench-loading">正在重新识别当前样本...</div>';
  } else if (result?.error) {
    content = `<div class="workbench-error">${escHtml(result.error)}</div>`;
  } else if (!result) {
    content =
      '<div class="workbench-empty">选择样本后点击「重新运行」</div>';
  }

  if (workbenchState.comparisonMode === "truth" && !hasTruthHtml) {
    content = `<div class="workbench-empty">无结构真值：${escHtml(
      groundTruth.reason || "数据集没有提供单元格或 HTML 真值",
    )}</div>`;
  } else if (result) {
    content += renderHtmlPayload(
      activePayload.html || "",
      workbenchState.predictionMode,
    );
  }
  if (!hasTruthHtml && workbenchState.comparisonMode === "prediction") {
    content += `<div class="workbench-meta">结构真值：${escHtml(
      groundTruth.reason || "不可用",
    )}</div>`;
  }

  const comparisons = `<button data-comparison="prediction" class="${workbenchState.comparisonMode === "prediction" ? "active" : ""}">预测</button>
    <button data-comparison="truth" class="${workbenchState.comparisonMode === "truth" ? "active" : ""}" ${hasTruthHtml ? "" : "disabled"}>真值</button>`;
  const modes = `<button data-mode="render" class="${workbenchState.predictionMode === "render" ? "active" : ""}">渲染</button>
    <button data-mode="source" class="${workbenchState.predictionMode === "source" ? "active" : ""}">源码</button>`;

  return `<div class="workbench-result">
    <div class="workbench-section-title">模型输出</div>
    <span class="workbench-status ${status}">${statusLabel(status)}</span>
    <div class="workbench-switches">${comparisons}${modes}</div>
    ${content}
  </div>`;
}

function renderToolbar() {
  const datasetOptions = workbenchState.datasets
    .map(
      (dataset) =>
        `<option value="${escHtml(dataset.id)}"${
          dataset.id === workbenchState.datasetId ? " selected" : ""
        }>${escHtml(dataset.name)}（${dataset.total_samples ?? "不可用"}）</option>`,
    )
    .join("");
  const samples = workbenchState.page?.samples || [];
  const sampleOptions = samples
    .map(
      (sample) =>
        `<option value="${escHtml(sample.sample_id)}"${
          sample.sample_id === workbenchState.selectedSampleId
            ? " selected"
            : ""
        }>#${sample.global_index} ${escHtml(sample.file_name)}</option>`,
    )
    .join("");
  const start = workbenchState.offset + 1;
  const end = workbenchState.offset + samples.length;

  return `<div class="workbench-toolbar">
    <select data-role="dataset">${datasetOptions}</select>
    <button data-action="prev" ${workbenchState.offset <= 0 ? "disabled" : ""}>上一页</button>
    <span>${samples.length ? `${start}-${end}` : "0"}</span>
    <button data-action="next" ${workbenchState.page?.has_more ? "" : "disabled"}>下一页</button>
    <select data-role="sample">${sampleOptions}</select>
    <button data-action="run" ${workbenchState.selectedSampleId && !workbenchState.running ? "" : "disabled"}>重新运行</button>
    <div class="workbench-id">${escHtml(workbenchState.selectedSampleId || "")}</div>
  </div>`;
}

function renderWorkbench() {
  const view = document.getElementById("workbenchView");
  if (workbenchState.error) {
    view.innerHTML = `<div class="workbench-error">${escHtml(workbenchState.error)}</div>`;
    return;
  }
  if (workbenchState.loading) {
    view.innerHTML =
      '<div class="workbench-loading">正在加载数据集样本...</div>';
    return;
  }
  if (!workbenchState.datasets.length) {
    view.innerHTML =
      '<div class="workbench-empty">没有已注册的数据集</div>';
    return;
  }

  const sample = selectedSample();
  view.innerHTML =
    renderToolbar()
    + '<div class="workbench-layout">'
    + renderPreview(sample, workbenchState.runResult)
    + renderResult(sample, workbenchState.runResult)
    + "</div>";
}

async function loadSamples() {
  if (!workbenchState.datasetId) return;
  workbenchState.loading = true;
  workbenchState.error = "";
  workbenchState.runResult = null;
  renderWorkbench();
  try {
    workbenchState.page = await listDatasetSamples(
      workbenchState.datasetId,
      workbenchState.offset,
      workbenchState.limit,
    );
    const samples = workbenchState.page.samples || [];
    workbenchState.selectedSampleId =
      samples[0]?.sample_id || null;
  } catch (error) {
    workbenchState.error = `样本加载失败: ${error.message}`;
  } finally {
    workbenchState.loading = false;
    renderWorkbench();
  }
}

async function loadDatasets() {
  workbenchState.loading = true;
  workbenchState.error = "";
  renderWorkbench();
  try {
    const payload = await listDatasets();
    workbenchState.datasets = payload.datasets || [];
    const available = workbenchState.datasets.find(
      (dataset) => dataset.available,
    );
    if (!available) {
      const detail = workbenchState.datasets
        .map((dataset) => `${dataset.id}: ${dataset.error}`)
        .join("\n");
      throw new Error(detail || "没有可用数据集");
    }
    workbenchState.datasetId = available.id;
    workbenchState.offset = 0;
    workbenchState.initialized = true;
    workbenchState.loading = false;
    await loadSamples();
  } catch (error) {
    workbenchState.error =
      `数据集不可用: ${error.message}。可设置 PP_FINTABNET_PATH`;
    workbenchState.loading = false;
    renderWorkbench();
  }
}

async function ensureWorkbench() {
  if (!workbenchState.initialized) {
    await loadDatasets();
  } else {
    renderWorkbench();
  }
}

async function runSelectedSample() {
  if (!workbenchState.selectedSampleId) return;
  workbenchState.running = true;
  workbenchState.error = "";
  renderWorkbench();
  try {
    workbenchState.runResult = await runDatasetSample(
      workbenchState.datasetId,
      workbenchState.selectedSampleId,
    );
    workbenchState.comparisonMode = "prediction";
  } catch (error) {
    workbenchState.runResult = {
      ok: false,
      status: "failed",
      error: `运行请求失败: ${error.message}`,
      ground_truth: selectedSample()?.ground_truth || {},
    };
  } finally {
    workbenchState.running = false;
    renderWorkbench();
  }
}

function handleClick(event) {
  const action = event.target.closest("[data-action]");
  if (action) {
    if (action.dataset.action === "prev") {
      workbenchState.offset = Math.max(
        0,
        workbenchState.offset - workbenchState.limit,
      );
      loadSamples();
    } else if (action.dataset.action === "next") {
      workbenchState.offset += workbenchState.limit;
      loadSamples();
    } else if (action.dataset.action === "run") {
      runSelectedSample();
    }
    return;
  }
  const comparison = event.target.closest("[data-comparison]");
  if (comparison && !comparison.disabled) {
    workbenchState.comparisonMode = comparison.dataset.comparison;
    renderWorkbench();
    return;
  }
  const mode = event.target.closest("[data-mode]");
  if (mode) {
    workbenchState.predictionMode = mode.dataset.mode;
    renderWorkbench();
  }
}

function handleChange(event) {
  if (event.target.dataset.role === "dataset") {
    workbenchState.datasetId = event.target.value;
    workbenchState.offset = 0;
    loadSamples();
  } else if (event.target.dataset.role === "sample") {
    workbenchState.selectedSampleId = event.target.value;
    workbenchState.runResult = null;
    workbenchState.comparisonMode = "prediction";
    renderWorkbench();
  }
}

export function initWorkbench() {
  const view = document.getElementById("workbenchView");
  view.addEventListener("click", handleClick);
  view.addEventListener("change", handleChange);
  window.addEventListener("pp:workbench-open", ensureWorkbench);
}
