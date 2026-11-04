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

try {
  const scope = await server.ssrLoadModule('/src/trainingDraftScope.ts');
  const snapshot = (templateId, name = templateId) => ({
    request: { name, node_id: 'node', profile_id: 'profile', template_id: templateId,
      inputs: {}, parameters: { batch_size: 16 }, gpu_count: templateId === 'openpi-norm' ? 0 : 1 },
    step: 0, advancedRecipe: false, editedParameters: [], editedStageParameters: [], entrypointEdited: false,
  });
  const pending = (templateId, name, writer = 'writer') => ({
    revision: 7, writer_id: writer, content: scope.encodeScopedTrainingDraft(snapshot(templateId, name), templateId),
  });
  const storage = () => {
    const entries = new Map();
    return { entries, getItem: (key) => entries.get(key) ?? null,
      setItem: (key, value) => entries.set(key, value), removeItem: (key) => entries.delete(key) };
  };
  const read = (store, user, template) => scope.readTrainingDraftPending(store, user, template);
  const legacyKey = 'termitech.training-draft.pending.v1:alice';

  await test('drafts for SFT, norm and ReCap coexist and clearing one leaves other templates and accounts intact', () => {
    const store = storage();
    for (const user of ['alice', 'bob'])
      for (const template of ['openpi-sft', 'openpi-norm', 'recap'])
        scope.writeTrainingDraftPending(store, user, template, pending(template, `${user}-${template}`));
    for (const user of ['alice', 'bob'])
      for (const template of ['openpi-sft', 'openpi-norm', 'recap'])
        assert.equal(scope.decodeScopedTrainingDraft(read(store, user, template).value.content, template).request.name, `${user}-${template}`);
    scope.clearTrainingDraftPending(store, 'alice', 'openpi-norm', 'writer');
    assert.equal(read(store, 'alice', 'openpi-norm'), undefined);
    assert.equal(store.entries.size, 5);
    assert.equal(read(store, 'alice', 'openpi-sft').value.revision, 7);
    assert.ok(read(store, 'bob', 'openpi-norm'));
  });

  await test('API scope and storage identity safely encode both account and template', () => {
    assert.equal(scope.trainingDraftPath('a&template_id=b'), '/training-draft?template_id=a%26template_id%3Db');
    assert.notEqual(scope.trainingDraftPendingKey('a:b', 'c'), scope.trainingDraftPendingKey('a', 'b:c'));
    assert.notEqual(scope.trainingDraftPendingKey('a', 'openpi-sft'), scope.trainingDraftPendingKey('a', 'openpi-norm'));
  });

  await test('old services that ignore the template query or echo another template are rejected', () => {
    assert.throws(() => scope.requireTrainingDraftScope({ content: null, revision: 0 }, 'openpi-sft'), /更新管理服务/);
    assert.throws(() => scope.requireTrainingDraftScope({ template_id: 'openpi-norm' }, 'openpi-sft'));
    assert.doesNotThrow(() => scope.requireTrainingDraftScope({ template_id: 'openpi-sft' }, 'openpi-sft'));
  });

  await test('legacy unsynchronised inputs are restored only into their original template and migrated after a scoped write', () => {
    const store = storage();
    const old = { ...pending('openpi-sft', 'unsynced sft'), user_id: 'alice' };
    store.setItem(legacyKey, JSON.stringify(old));
    assert.equal(read(store, 'alice', 'openpi-norm'), undefined);
    assert.equal(read(store, 'bob', 'openpi-sft'), undefined);
    const adopted = read(store, 'alice', 'openpi-sft');
    assert.equal(adopted.value.revision, old.revision);
    assert.equal(adopted.key, legacyKey);
    scope.writeTrainingDraftPending(store, 'alice', 'openpi-sft', pending('openpi-sft', 'continued sft'), adopted);
    assert.equal(store.getItem(legacyKey), null);
    assert.equal(read(store, 'alice', 'openpi-sft').value.template_id, 'openpi-sft');
    assert.equal(scope.decodeScopedTrainingDraft(read(store, 'alice', 'openpi-sft').value.content, 'openpi-sft').request.name, 'continued sft');
  });

  await test('a different tab replacing the old shared checkpoint cannot lose its new template during migration or clear', () => {
    const store = storage();
    store.setItem(legacyKey, JSON.stringify({ ...pending('openpi-sft', 'sft'), user_id: 'alice' }));
    const adopted = read(store, 'alice', 'openpi-sft');
    const other = JSON.stringify({ ...pending('openpi-norm', 'new norm', 'other'), user_id: 'alice' });
    store.setItem(legacyKey, other);
    scope.writeTrainingDraftPending(store, 'alice', 'openpi-sft', pending('openpi-sft', 'sft'), adopted);
    assert.equal(store.getItem(legacyKey), other);
    scope.clearTrainingDraftPending(store, 'alice', 'openpi-sft', 'writer', adopted);
    assert.equal(store.getItem(legacyKey), other);
    assert.equal(read(store, 'alice', 'openpi-norm').value.writer_id, 'other');
  });

  await test('clearing an adopted checkpoint respects a newer writer in the same template', () => {
    const store = storage();
    scope.writeTrainingDraftPending(store, 'alice', 'recap', pending('recap', 'old', 'old-writer'));
    const adopted = read(store, 'alice', 'recap');
    scope.writeTrainingDraftPending(store, 'alice', 'recap', pending('recap', 'new', 'new-writer'));
    scope.clearTrainingDraftPending(store, 'alice', 'recap', 'old-writer', adopted);
    assert.equal(read(store, 'alice', 'recap').value.writer_id, 'new-writer');
    scope.clearTrainingDraftPending(store, 'alice', 'recap', 'new-writer');
    assert.equal(read(store, 'alice', 'recap'), undefined);
  });

  await test('mismatched template content or ownership cannot be encoded, written or restored', () => {
    const store = storage();
    const sft = pending('openpi-sft', 'sft');
    assert.equal(scope.decodeScopedTrainingDraft(sft.content, 'openpi-norm'), undefined);
    assert.throws(() => scope.encodeScopedTrainingDraft(snapshot('openpi-sft'), 'openpi-norm'));
    assert.throws(() => scope.writeTrainingDraftPending(store, 'alice', 'openpi-norm', sft));
    assert.equal(store.entries.size, 0);
    const key = scope.trainingDraftPendingKey('alice', 'openpi-norm');
    for (const value of [
      { ...sft, user_id: 'alice', template_id: 'openpi-norm' },
      { ...pending('openpi-norm', 'norm'), user_id: 'bob', template_id: 'openpi-norm' },
      { ...pending('openpi-norm', 'norm'), user_id: 'alice', template_id: 'recap' },
    ]) {
      store.setItem(key, JSON.stringify(value));
      assert.equal(read(store, 'alice', 'openpi-norm'), undefined);
    }
  });

  await test('failure to write the scoped checkpoint preserves the adopted legacy recovery copy', () => {
    const store = storage();
    const raw = JSON.stringify({ ...pending('openpi-sft', 'sft'), user_id: 'alice' });
    store.setItem(legacyKey, raw);
    const adopted = read(store, 'alice', 'openpi-sft');
    const blocked = { ...store, setItem() { throw new Error('quota'); } };
    assert.throws(() => scope.writeTrainingDraftPending(blocked, 'alice', 'openpi-sft', pending('openpi-sft', 'new'), adopted));
    assert.equal(store.getItem(legacyKey), raw);
  });
} finally {
  await server.close();
}
