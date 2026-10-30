import { useEffect, useRef, useState } from 'react';
import type { CSSProperties, KeyboardEvent, PointerEvent, RefObject } from 'react';
import { fitDialogBounds, resizeDialogBounds } from './dialogResizeBounds';
import type { DialogBounds, DialogEdge } from './dialogResizeBounds';
import './resizable-dialog.css';

const edges: DialogEdge[] = ['n', 'e', 's', 'w', 'ne', 'nw', 'sw', 'se'];
const viewport = () => ({ width: window.innerWidth, height: window.innerHeight });

export function useDialogResize(ref: RefObject<HTMLDivElement | null>) {
  const [bounds, setBounds] = useState<DialogBounds | null>(null);
  const drag = useRef<{
    pointerId: number;
    edge: DialogEdge;
    x: number;
    y: number;
    initial: DialogBounds;
  } | null>(null);

  useEffect(() => {
    const onViewportResize = () => {
      drag.current = null;
      setBounds((current) => (current ? fitDialogBounds(current, viewport()) : null));
    };
    window.addEventListener('resize', onViewportResize);
    return () => {
      window.removeEventListener('resize', onViewportResize);
      drag.current = null;
    };
  }, []);

  const start = (event: PointerEvent<HTMLButtonElement>, edge: DialogEdge) => {
    if (event.button !== 0 || !event.isPrimary || !ref.current) return;
    event.preventDefault();
    event.stopPropagation();
    const initial = ref.current.getBoundingClientRect();
    drag.current = {
      pointerId: event.pointerId,
      edge,
      x: event.clientX,
      y: event.clientY,
      initial,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const move = (event: PointerEvent<HTMLButtonElement>) => {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    event.preventDefault();
    setBounds(
      resizeDialogBounds(
        current.initial,
        current.edge,
        event.clientX - current.x,
        event.clientY - current.y,
        viewport(),
      ),
    );
  };

  const end = (event: PointerEvent<HTMLButtonElement>) => {
    if (drag.current?.pointerId !== event.pointerId) return;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const keyboardResize = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.altKey || event.metaKey || event.ctrlKey || !ref.current) return;
    const steps: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    };
    const step = steps[event.key];
    if (!step) return;
    event.preventDefault();
    event.stopPropagation();
    const size = event.shiftKey ? 40 : 10;
    setBounds(
      resizeDialogBounds(
        ref.current.getBoundingClientRect(),
        'se',
        step[0] * size,
        step[1] * size,
        viewport(),
      ),
    );
  };

  const style: CSSProperties | undefined = bounds
    ? { position: 'absolute', ...bounds, maxWidth: 'none', maxHeight: 'none', margin: 0 }
    : undefined;

  const handles = edges.map((edge) => (
    <button
      key={edge}
      type="button"
      className={`dialog-resize-handle dialog-resize-${edge}`}
      aria-label={edge === 'se' ? '调整弹窗大小（方向键调整，双击恢复默认）' : undefined}
      aria-hidden={edge !== 'se' ? true : undefined}
      tabIndex={edge === 'se' ? 0 : -1}
      title="拖动边框或四角调整大小；双击右下角恢复默认"
      onPointerDown={(event) => start(event, edge)}
      onPointerMove={move}
      onPointerUp={end}
      onPointerCancel={end}
      onLostPointerCapture={() => {
        drag.current = null;
      }}
      onKeyDown={edge === 'se' ? keyboardResize : undefined}
      onDoubleClick={edge === 'se' ? () => setBounds(null) : undefined}
    />
  ));
  return { style, handles };
}
