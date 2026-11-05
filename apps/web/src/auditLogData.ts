import { ApiError, createOperationClient } from './api';
import { auditEventsPath, sanitizeAuditEntry } from './auditLogPresentation';
import type { AuditEventsPage, AuditFilters } from './auditLogPresentation';

export interface AuditPaging {
  cursor: string | null;
  previous: (string | null)[];
}
export const firstAuditPage = (): AuditPaging => ({ cursor: null, previous: [] });
export function nextAuditPage(paging: AuditPaging, page: AuditEventsPage): AuditPaging {
  if (!page.has_more || !page.next_cursor) return paging;
  if (page.next_cursor === paging.cursor || paging.previous.includes(page.next_cursor))
    throw new Error('分页游标未推进，请刷新首页后重试。');
  return { cursor: page.next_cursor, previous: [...paging.previous, paging.cursor] };
}
export function previousAuditPage(paging: AuditPaging): AuditPaging {
  if (!paging.previous.length) return paging;
  return { cursor: paging.previous.at(-1)!, previous: paging.previous.slice(0, -1) };
}

/** Read requests are bound to the opening account and latest requested page. */
export function createAuditEventsClient() {
  const client = createOperationClient();
  let generation = 0;
  let controller: AbortController | undefined;
  let disposed = false;
  const cancel = () => {
    generation++;
    controller?.abort();
  };
  return {
    cancel,
    dispose() {
      disposed = true;
      cancel();
    },
    async load(filters: AuditFilters, cursor: string | null = null): Promise<AuditEventsPage> {
      if (disposed) throw new DOMException('审计窗口已关闭。', 'AbortError');
      client.requireCurrent();
      cancel();
      const current = generation;
      const abort = new AbortController();
      controller = abort;
      const page = await client.api<AuditEventsPage>(auditEventsPath(filters, cursor), {
        signal: AbortSignal.any([abort.signal, AbortSignal.timeout(20000)]),
        cache: 'no-store',
      });
      if (disposed || current !== generation || abort.signal.aborted)
        throw new DOMException('审计查询已改变。', 'AbortError');
      if (
        !page ||
        !Array.isArray(page.items) ||
        typeof page.has_more !== 'boolean' ||
        (page.next_cursor !== null && typeof page.next_cursor !== 'string') ||
        (page.has_more && !page.next_cursor) ||
        !page.items.every(
          (entry) =>
            entry &&
            typeof entry.id === 'string' &&
            typeof entry.created_at === 'string' &&
            Number.isInteger(entry.status) &&
            entry.status >= 100 &&
            entry.status <= 599,
        )
      )
        throw new ApiError('INVALID_RESPONSE', '审计分页响应无效，请重试。', 0);
      return {
        items: page.items.map(sanitizeAuditEntry),
        next_cursor: page.next_cursor,
        has_more: page.has_more,
      };
    },
  };
}
