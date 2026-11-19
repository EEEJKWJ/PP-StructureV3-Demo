async function responseJson(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`接口返回了无效 JSON（HTTP ${response.status}）`);
  }
  if (!response.ok) {
    throw new Error(
      data?.detail
      || data?.error
      || `接口请求失败（HTTP ${response.status}）`,
    );
  }
  return data;
}

export async function getModelStatus() {
  return responseJson(await fetch("/api/model-status"));
}

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  return responseJson(
    await fetch("/api/upload", { method: "POST", body: form }),
  );
}

export async function getTextSource(docId, page) {
  return responseJson(
    await fetch(`/api/text-source/${docId}/${page}`),
  );
}

export async function detectPage(docId, page) {
  return responseJson(
    await fetch(`/api/detect/${docId}/${page}`, {
      method: "POST",
    }),
  );
}

export async function detectAllPages(docId) {
  return responseJson(
    await fetch(`/api/detect-all/${docId}`, {
      method: "POST",
    }),
  );
}

export async function listModels() {
  return responseJson(await fetch("/api/models"));
}

export async function activateModel(modelId) {
  return responseJson(
    await fetch(
      `/api/models/${encodeURIComponent(modelId)}/activate`,
      { method: "POST" },
    ),
  );
}

export async function listEvaluations() {
  return responseJson(await fetch("/api/evaluations"));
}

export async function latestEvaluationMetrics() {
  return responseJson(
    await fetch("/api/evaluations/latest/metrics"),
  );
}

export async function listDatasets() {
  return responseJson(await fetch("/api/datasets"));
}

export async function listDatasetSamples(
  datasetId,
  offset = 0,
  limit = 20,
) {
  const query = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
  });
  return responseJson(
    await fetch(
      `/api/datasets/${encodeURIComponent(datasetId)}/samples?${query}`,
    ),
  );
}

export async function runDatasetSample(datasetId, sampleId) {
  return responseJson(
    await fetch(
      `/api/datasets/${encodeURIComponent(datasetId)}/samples/`
        + `${encodeURIComponent(sampleId)}/run`,
      { method: "POST" },
    ),
  );
}

export function datasetSampleImageUrl(datasetId, sampleId) {
  return `/api/datasets/${encodeURIComponent(datasetId)}/samples/`
    + `${encodeURIComponent(sampleId)}/image`;
}

export async function listComparisons() {
  return responseJson(await fetch("/api/comparisons"));
}

export async function getComparison(comparisonId) {
  return responseJson(
    await fetch(
      `/api/comparisons/${encodeURIComponent(comparisonId)}`,
    ),
  );
}

export function comparisonExportUrl(comparisonId, format) {
  return `/api/comparisons/${encodeURIComponent(comparisonId)}`
    + `/export/${encodeURIComponent(format)}`;
}
