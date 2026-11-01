import type { RunEvent } from './types';

export interface TrainingMetricPage {
  attempt_id: string | null;
  items: RunEvent[];
  next_seq: number;
  has_more: boolean;
  total_count: number;
}

export interface TrainingMetricFeed {
  items: RunEvent[];
  cursor: number;
  totalCount: number;
}

export function appendTrainingMetricPage(
  previous: TrainingMetricFeed,
  page: TrainingMetricPage,
  attemptId: string,
): TrainingMetricFeed {
  if (
    page.attempt_id !== attemptId ||
    !Array.isArray(page.items) ||
    !Number.isSafeInteger(page.next_seq) ||
    page.next_seq < previous.cursor ||
    !Number.isSafeInteger(page.total_count) ||
    page.total_count < 0 ||
    typeof page.has_more !== 'boolean'
  ) {
    throw new Error('指标历史响应与当前训练执行不一致，请重新读取。');
  }
  let cursor = previous.cursor;
  for (const item of page.items) {
    if (
      item.attempt_id !== attemptId ||
      item.type !== 'METRIC' ||
      !Number.isSafeInteger(item.seq) ||
      item.seq <= cursor
    ) {
      throw new Error('指标历史顺序无效，请重新读取。');
    }
    cursor = item.seq;
  }
  if (page.next_seq !== cursor || (page.has_more && !page.items.length)) {
    throw new Error('指标历史分页未前进，请重新读取。');
  }
  return {
    // Keep the complete metric history. The separate log/event feed still has
    // a bounded tail; log volume must never evict early metric observations.
    items: [...previous.items, ...page.items],
    cursor,
    totalCount: page.total_count,
  };
}
