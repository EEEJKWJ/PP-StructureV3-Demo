import {
  comparisonExportUrl,
  getComparison,
  listComparisons,
} from "./api.js";
import { escHtml } from "./results.js";

const comparisonState = {
  initialized: false,
  loading: false,
  error: "",
  reports: [],
  selectedId: null,
  detail: null,
};

function formatNumber(value, digits = 2) {
  return value === null || value === undefined
    ? "不适用"
    : Number(value).toFixed(digits);
}

function renderModelTable(detail) {
  const body = (detail.models || [])
    .map((item) => {
      const model = item.model || {};
      const metrics = item.metrics || {};
      return `<tr>
        <td>${escHtml(model.name || model.id || "unknown")}</td>
        <td>${item.summary?.completed ?? 0}</td>
        <td>${item.summary?.successes ?? 0}</td>
        <td>${formatNumber(metrics.table_detection_rate)}</td>
        <td>${formatNumber(metrics.average_cells, 1)}</td>
        <td>${formatNumber(metrics.latency_p50_ms, 1)}</td>
        <td>${formatNumber(metrics.latency_p95_ms, 1)}</td>
      </tr>`;
    })
    .join("");
  return `<table class="comparison-table">
    <thead><tr><th>模型</th><th>完成</th><th>成功</th><th>检出率</th>
    <th>平均单元格</th><th>P50 ms</th><th>P95 ms</th></tr></thead>
    <tbody>${body}</tbody>
  </table>`;
}

function renderSampleTable(detail) {
  const modelIds = detail.model_ids || [];
  const headers = modelIds
    .map((modelId) => `<th>${escHtml(modelId)}</th>`)
    .join("");
  const rows = (detail.samples || [])
    .map((sample) => {
      const modelCells = modelIds
        .map((modelId) => {
          const result = sample.models[modelId] || {};
          return `<td>
            <span class="comparison-error ${escHtml(result.error_type || "")}">
              ${escHtml(result.error_type || "missing")}
            </span>
            <br>${result.cell_count ?? 0} cells ·
            ${formatNumber(result.elapsed_ms, 1)} ms
          </td>`;
        })
        .join("");
      return `<tr>
        <td>${sample.global_index ?? ""}</td>
        <td title="${escHtml(sample.sample_id || "")}">${escHtml(sample.file_name || "")}</td>
        ${modelCells}
        <td>${escHtml(sample.completion_winner || "—")}</td>
        <td>${escHtml(sample.quality_winner || "不可比较")}</td>
        <td>${escHtml(sample.latency_winner || "—")}</td>
      </tr>`;
    })
    .join("");
  return `<table class="comparison-table comparison-samples">
    <thead><tr><th>#</th><th>文件</th>${headers}
    <th>完成胜者</th><th>质量胜者</th><th>延迟胜者</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderDetail() {
  const detail = comparisonState.detail;
  if (!detail) return "";
  const summary = detail.comparison || {};
  const comparisonId = detail.comparison_id;
  return `<div class="comparison-summary">
    <span>样本 ${detail.sample_count}</span>
    <span>质量可比较 ${summary.quality_comparable_samples || 0}</span>
    <a href="${comparisonExportUrl(comparisonId, "json")}" download>JSON</a>
    <a href="${comparisonExportUrl(comparisonId, "csv")}" download>CSV</a>
    <a href="${comparisonExportUrl(comparisonId, "html")}" download>HTML</a>
  </div>
  ${renderModelTable(detail)}
  <div class="workbench-section-title">逐样本对比</div>
  ${renderSampleTable(detail)}`;
}

function renderComparison() {
  const view = document.getElementById("comparisonView");
  if (comparisonState.loading) {
    view.innerHTML =
      '<div class="workbench-loading">正在加载对比报告...</div>';
    return;
  }
  if (comparisonState.error) {
    view.innerHTML =
      `<div class="workbench-error">${escHtml(comparisonState.error)}</div>`;
    return;
  }
  if (!comparisonState.reports.length) {
    view.innerHTML = `<div class="workbench-empty">
      暂无模型对比报告。可运行：<br>
      <code>python scripts/compare_models.py --models pp-tablemagic-v2,rapidtable-slanet-plus --limit 10</code>
    </div>`;
    return;
  }
  const options = comparisonState.reports
    .map(
      (report) =>
        `<option value="${escHtml(report.comparison_id)}"${
          report.comparison_id === comparisonState.selectedId
            ? " selected"
            : ""
        }>${escHtml(report.comparison_id)} · ${escHtml(
          report.generated_at || "",
        )}</option>`,
    )
    .join("");
  view.innerHTML = `<div class="workbench-toolbar">
    <select data-role="comparison">${options}</select>
  </div>${renderDetail()}`;
}

async function loadDetail() {
  if (!comparisonState.selectedId) {
    comparisonState.detail = null;
    return;
  }
  comparisonState.detail = await getComparison(
    comparisonState.selectedId,
  );
}

async function loadReports() {
  comparisonState.loading = true;
  comparisonState.error = "";
  renderComparison();
  try {
    const payload = await listComparisons();
    comparisonState.reports = payload.reports || [];
    comparisonState.selectedId =
      comparisonState.reports[0]?.comparison_id || null;
    await loadDetail();
    comparisonState.initialized = true;
  } catch (error) {
    comparisonState.error = `对比报告加载失败: ${error.message}`;
  } finally {
    comparisonState.loading = false;
    renderComparison();
  }
}

async function ensureComparison() {
  if (!comparisonState.initialized) {
    await loadReports();
  } else {
    renderComparison();
  }
}

async function handleChange(event) {
  if (event.target.dataset.role !== "comparison") return;
  comparisonState.selectedId = event.target.value;
  comparisonState.loading = true;
  comparisonState.error = "";
  renderComparison();
  try {
    await loadDetail();
  } catch (error) {
    comparisonState.error = `报告详情加载失败: ${error.message}`;
  } finally {
    comparisonState.loading = false;
    renderComparison();
  }
}

export function initComparison() {
  const view = document.getElementById("comparisonView");
  view.addEventListener("change", handleChange);
  window.addEventListener("pp:comparisons-open", ensureComparison);
}
