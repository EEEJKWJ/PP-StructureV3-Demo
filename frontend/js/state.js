const CACHE_KEY = "pp-struct-state";

export const state = {
  docId: null,
  pages: [],
  activePage: null,
  detectedPages: {},
  isPdf: false,
  resultView: "visual",
  allPagesData: null,
};

export const runtime = {
  modelReady: false,
};

export function saveCache() {
  const {
    docId,
    pages,
    activePage,
    detectedPages,
    isPdf,
    resultView,
    allPagesData,
  } = state;
  localStorage.setItem(
    CACHE_KEY,
    JSON.stringify({
      docId,
      pages,
      activePage,
      detectedPages,
      isPdf,
      resultView,
      allPagesData,
    }),
  );
}

export function loadCache() {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return false;
    const cached = JSON.parse(raw);
    if (!cached.docId || !cached.pages?.length) return false;
    Object.assign(state, cached);
    return true;
  } catch {
    return false;
  }
}
