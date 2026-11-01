import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});
const event = (seq, attempt = 'attempt-one') => ({
  attempt_id: attempt,
  seq,
  type: 'METRIC',
  data: { step: seq - 1, loss: seq === 2 ? null : 1 / seq },
});
const page = (items, changes = {}) => ({
  attempt_id: 'attempt-one',
  items,
  next_seq: items.at(-1)?.seq ?? 0,
  has_more: false,
  total_count: items.length,
  ...changes,
});
const empty = () => ({ items: [], cursor: 0, totalCount: 0 });

try {
  const { appendTrainingMetricPage } = await server.ssrLoadModule('/src/trainingMetricFeed.ts');
  await test('metric-only pagination retains the beginning beyond the log tail limit', () => {
    let feed = empty();
    for (let offset = 0; offset < 6000; offset += 1000) {
      const items = Array.from({ length: 1000 }, (_, i) => event(offset + i + 1));
      feed = appendTrainingMetricPage(
        feed,
        page(items, { total_count: 6000, has_more: offset < 5000 }),
        'attempt-one',
      );
    }
    assert.equal(feed.items.length, 6000);
    assert.equal(feed.items[0].data.step, 0);
    assert.equal(feed.items[1].data.loss, null);
    assert.equal(feed.items.at(-1).data.step, 5999);
    assert.equal(feed.cursor, 6000);
    assert.equal(feed.totalCount, 6000);
  });
  await test('an idle live read preserves history and cursor without fabricating observations', () => {
    const initial = appendTrainingMetricPage(empty(), page([event(31)]), 'attempt-one');
    const next = appendTrainingMetricPage(
      initial,
      page([], { next_seq: 31, total_count: 1 }),
      'attempt-one',
    );
    assert.deepEqual(next, initial);
    assert.equal(initial.items.length, 1);
  });
  await test('smaller pages with has_more are valid and preserve sparse metric reports', () => {
    const first = appendTrainingMetricPage(
      empty(),
      page([event(5)], { has_more: true, total_count: 2 }),
      'attempt-one',
    );
    const sparse = { ...event(50), data: { step: 1200, 'eval/score': 0 } };
    const final = appendTrainingMetricPage(
      first,
      page([sparse], { total_count: 2 }),
      'attempt-one',
    );
    assert.deepEqual(
      final.items.map((item) => item.seq),
      [5, 50],
    );
    assert.deepEqual(final.items[1].data, { step: 1200, 'eval/score': 0 });
  });
  await test('another execution cannot contaminate the active execution history', () => {
    assert.throws(() =>
      appendTrainingMetricPage(empty(), page([event(1)], { attempt_id: 'other' }), 'attempt-one'),
    );
    assert.throws(() =>
      appendTrainingMetricPage(empty(), page([event(1, 'other')]), 'attempt-one'),
    );
  });
  await test('invalid and nonadvancing pages cannot silently drop history or spin indefinitely', () => {
    const invalid = [
      page([], { has_more: true }),
      page([event(2), event(1)]),
      page([{ ...event(1), type: 'LOG' }]),
      page([event(1)], { next_seq: 10 }),
      page([event(1)], { total_count: -1 }),
      page([event(1)], { next_seq: NaN }),
    ];
    for (const value of invalid)
      assert.throws(() => appendTrainingMetricPage(empty(), value, 'attempt-one'));
    const prior = appendTrainingMetricPage(empty(), page([event(10)]), 'attempt-one');
    assert.throws(() => appendTrainingMetricPage(prior, page([event(10)]), 'attempt-one'));
    assert.throws(() => appendTrainingMetricPage(prior, page([], { next_seq: 0 }), 'attempt-one'));
    assert.equal(prior.items.length, 1);
  });
} finally {
  await server.close();
}
