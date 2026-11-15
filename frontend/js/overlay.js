import {
  highlightBlock,
  highlightCell,
  highlightTocItem,
} from "./interactions.js";
import { LABEL_COLORS } from "./theme.js";

export function drawBoxes(data) {
  const container = document.getElementById("imgContainer");
  const image = document.getElementById("pageImg");
  if (!container || !image) return;

  container
    .querySelectorAll(".bbox-overlay, .cell-overlay, .toc-overlay")
    .forEach((element) => element.remove());

  const scaleX = image.clientWidth / data.width;
  const scaleY = image.clientHeight / data.height;

  data.blocks.forEach((block, blockIndex) => {
    const [x1, y1, x2, y2] = block.bbox;
    const color = LABEL_COLORS[block.label] || "#78909c";

    const box = document.createElement("div");
    box.className = "bbox-overlay";
    box.dataset.block = blockIndex;
    box.style.left = x1 * scaleX + "px";
    box.style.top = y1 * scaleY + "px";
    box.style.width = (x2 - x1) * scaleX + "px";
    box.style.height = (y2 - y1) * scaleY + "px";
    box.style.borderColor = color;
    box.style.background = color + "15";
    box.addEventListener("click", (event) => {
      event.stopPropagation();
      highlightBlock(blockIndex);
    });

    const label = document.createElement("span");
    label.className = "bbox-label";
    label.style.background = color;
    label.textContent = `${block.label} #${block.order}`;
    box.appendChild(label);
    container.appendChild(box);

    if (block.toc_items && block.toc_items.length > 0) {
      for (let tocIndex = 0; tocIndex < block.toc_items.length; tocIndex++) {
        const item = block.toc_items[tocIndex];
        if (!item.bbox) continue;
        const [tx1, ty1, tx2, ty2] = item.bbox;
        const tocBox = document.createElement("div");
        tocBox.className = "toc-overlay";
        tocBox.dataset.block = blockIndex;
        tocBox.dataset.toc = tocIndex;
        tocBox.style.left = tx1 * scaleX + "px";
        tocBox.style.top = ty1 * scaleY + "px";
        tocBox.style.width = (tx2 - tx1) * scaleX + "px";
        tocBox.style.height = (ty2 - ty1) * scaleY + "px";
        tocBox.addEventListener("click", (event) => {
          event.stopPropagation();
          highlightTocItem(blockIndex, tocIndex);
        });
        container.appendChild(tocBox);
      }
    }

    if (block.cells && block.cells.length > 0) {
      for (const cell of block.cells) {
        const [cx1, cy1, cx2, cy2] = cell.bbox;
        const cellBox = document.createElement("div");
        cellBox.className = "cell-overlay";
        cellBox.dataset.block = blockIndex;
        cellBox.dataset.row = cell.row;
        cellBox.dataset.col = cell.col;
        cellBox.style.left = cx1 * scaleX + "px";
        cellBox.style.top = cy1 * scaleY + "px";
        cellBox.style.width = (cx2 - cx1) * scaleX + "px";
        cellBox.style.height = (cy2 - cy1) * scaleY + "px";
        cellBox.addEventListener("click", (event) => {
          event.stopPropagation();
          highlightCell(blockIndex, cell.row, cell.col);
        });
        container.appendChild(cellBox);
      }
    }
  });
}
