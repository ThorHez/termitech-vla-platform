interface ActivityTarget {
  addEventListener: (
    type: string,
    listener: EventListener,
    options?: AddEventListenerOptions,
  ) => void;
  removeEventListener: (type: string, listener: EventListener) => void;
}
interface ActivityClock {
  now: () => number;
  setTimeout: (callback: () => void, delay: number) => ReturnType<typeof setTimeout>;
  clearTimeout: (timer: ReturnType<typeof setTimeout>) => void;
}
const activityEvents = ['pointerdown', 'keydown', 'touchstart', 'wheel'];

// A timer or a successful API poll is never evidence of human activity. A
// trailing heartbeat preserves the last interaction when inputs arrive in bursts.
export function watchHumanActivity(
  onActivity: () => void,
  target: ActivityTarget = window,
  clock: ActivityClock = {
    now: () => Date.now(),
    setTimeout: (callback, delay) => setTimeout(callback, delay),
    clearTimeout: (timer) => clearTimeout(timer),
  },
  throttleMs = 30000,
) {
  let lastSent: number | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const send = () => {
    timer = undefined;
    lastSent = clock.now();
    onActivity();
  };
  const handler: EventListener = (event) => {
    if (!event.isTrusted) return;
    const remaining = lastSent === undefined ? 0 : throttleMs - (clock.now() - lastSent);
    if (remaining <= 0) {
      if (timer !== undefined) clock.clearTimeout(timer);
      send();
    } else if (timer === undefined) timer = clock.setTimeout(send, remaining);
  };
  for (const type of activityEvents) target.addEventListener(type, handler, { passive: true });
  return () => {
    for (const type of activityEvents) target.removeEventListener(type, handler);
    if (timer !== undefined) clock.clearTimeout(timer);
  };
}
