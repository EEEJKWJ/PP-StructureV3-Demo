import { toggleCells } from "./results.js";

export function clearHighlights() {
  document
    .querySelectorAll(".highlight")
    .forEach((element) => element.classList.remove("highlight"));
}

export function highlightBlock(blockIndex) {
  clearHighlights();
  const overlay = document.querySelector(
    `.bbox-overlay[data-block="${blockIndex}"]`,
  );
  if (overlay) {
    overlay.classList.add("highlight");
    overlay.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }
  const item = document.querySelector(
    `.block-item[data-block="${blockIndex}"]`,
  );
  if (item) {
    item.classList.add("highlight");
    item.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }
}

export function highlightCell(blockIndex, row, col) {
  clearHighlights();
  const cellElement = document.querySelector(
    `.cell-overlay[data-block="${blockIndex}"][data-row="${row}"][data-col="${col}"]`,
  );
  if (cellElement) {
    cellElement.classList.add("highlight");
    const preview = document.getElementById("preview");
    const rect = cellElement.getBoundingClientRect();
    const previewRect = preview.getBoundingClientRect();
    if (
      rect.top < previewRect.top
      || rect.bottom > previewRect.bottom
    ) {
      cellElement.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }

  const selector =
    `[data-block="${blockIndex}"][data-row="${row}"][data-col="${col}"]`;
  const target = document.querySelector(
    `.cells-table td${selector}, .tree-label${selector}, .tree-leaf${selector}`,
  );
  if (target) {
    target.classList.add("highlight");
    target.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
    const wrapper = target.closest('[id^="cells-"]');
    if (wrapper && wrapper.style.display === "none") {
      toggleCells(wrapper.id);
    }
  }
}

export function highlightScanCell(blockIndex, cellIndex) {
  clearHighlights();
  const overlay = document.querySelector(
    `.cell-overlay[data-block="${blockIndex}"][data-scancell="${cellIndex}"]`,
  );
  if (overlay) {
    overlay.classList.add("highlight");
    const preview = document.getElementById("preview");
    const rect = overlay.getBoundingClientRect();
    const previewRect = preview.getBoundingClientRect();
    if (
      rect.top < previewRect.top
      || rect.bottom > previewRect.bottom
    ) {
      overlay.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }

  const cell = document.querySelector(
    `.scan-table td[data-block="${blockIndex}"][data-scancell="${cellIndex}"]`,
  );
  if (cell) {
    cell.classList.add("highlight");
    cell.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
    const wrapper = cell.closest('[id^="scan-table-"]');
    if (wrapper && wrapper.style.display === "none") {
      toggleCells(wrapper.id);
    }
  }
}

export function highlightTocItem(blockIndex, tocIndex) {
  clearHighlights();
  const overlay = document.querySelector(
    `.toc-overlay[data-block="${blockIndex}"][data-toc="${tocIndex}"]`,
  );
  if (overlay) {
    overlay.classList.add("highlight");
    const preview = document.getElementById("preview");
    const rect = overlay.getBoundingClientRect();
    const previewRect = preview.getBoundingClientRect();
    if (
      rect.top < previewRect.top
      || rect.bottom > previewRect.bottom
    ) {
      overlay.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }

  const item = document.querySelector(
    `.toc-item[data-block="${blockIndex}"][data-toc="${tocIndex}"]`,
  );
  if (item) {
    item.classList.add("highlight");
    item.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
    const wrapper = item.closest('[id^="toc-"]');
    if (wrapper && wrapper.style.display === "none") {
      toggleCells(wrapper.id);
    }
  }
}
