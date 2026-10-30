export type DialogBounds = { left: number; top: number; width: number; height: number };
export type DialogViewport = { width: number; height: number };
export type DialogEdge = 'n' | 'e' | 's' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(Math.max(value, minimum), maximum);

function limits(viewport: DialogViewport) {
  const gap = Math.min(12, viewport.width / 4, viewport.height / 4);
  const width = viewport.width - gap * 2;
  const height = viewport.height - gap * 2;
  return { gap, width, height, minWidth: Math.min(360, width), minHeight: Math.min(240, height) };
}

export function fitDialogBounds(bounds: DialogBounds, viewport: DialogViewport): DialogBounds {
  const {
    gap,
    width: availableWidth,
    height: availableHeight,
    minWidth,
    minHeight,
  } = limits(viewport);
  const width = clamp(bounds.width, Math.min(bounds.width, minWidth), availableWidth);
  const height = clamp(bounds.height, Math.min(bounds.height, minHeight), availableHeight);
  return {
    width,
    height,
    left: clamp(bounds.left, gap, viewport.width - gap - width),
    top: clamp(bounds.top, gap, viewport.height - gap - height),
  };
}

export function resizeDialogBounds(
  initial: DialogBounds,
  edge: DialogEdge,
  deltaX: number,
  deltaY: number,
  viewport: DialogViewport,
): DialogBounds {
  const bounds = fitDialogBounds(initial, viewport);
  const limitsForViewport = limits(viewport);
  const { gap } = limitsForViewport;
  const minWidth = Math.min(bounds.width, limitsForViewport.minWidth);
  const minHeight = Math.min(bounds.height, limitsForViewport.minHeight);
  let { left, top } = bounds;
  let right = left + bounds.width;
  let bottom = top + bounds.height;
  if (edge.includes('w')) left = clamp(left + deltaX, gap, right - minWidth);
  if (edge.includes('e')) right = clamp(right + deltaX, left + minWidth, viewport.width - gap);
  if (edge.includes('n')) top = clamp(top + deltaY, gap, bottom - minHeight);
  if (edge.includes('s')) bottom = clamp(bottom + deltaY, top + minHeight, viewport.height - gap);
  return { left, top, width: right - left, height: bottom - top };
}
