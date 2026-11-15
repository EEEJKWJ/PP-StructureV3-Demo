async function responseJson(response) {
  return response.json();
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
