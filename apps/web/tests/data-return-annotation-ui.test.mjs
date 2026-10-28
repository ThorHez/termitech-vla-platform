import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});
const freeze = (value) => {
  if (value && typeof value === 'object') {
    for (const item of Object.values(value)) freeze(item);
    Object.freeze(value);
  }
  return value;
};
const buttonMarkup = (html) =>
  [...html.matchAll(/<button\b[^>]*>[\s\S]*?<\/button>/g)].map((match) => ({
    html: match[0],
    text: /aria-label="([^"]+)"/.exec(match[0])?.[1] || match[0].replace(/<[^>]*>/g, '').trim(),
  }));
const render = (element) => renderToStaticMarkup(element);
const noExternalMedia = (html) => assert.doesNotMatch(html, /<(?:iframe|video|img)\b[^>]*\bsrc=/);

try {
  const { default: Editor, DataReturnAnnotationSummary: Summary } = await server.ssrLoadModule(
    '/src/DataReturnAnnotationEditor.tsx',
  );
  const { createDataReturnSimulation } = await server.ssrLoadModule('/src/dataReturnSimulation.ts');
  const { default: DataReturnMock } = await server.ssrLoadModule('/src/DataReturnMock.tsx');
  const simulation = createDataReturnSimulation(
    { deploymentRecordId: 'record-label-preview', name: '标注演示部署', machine: '模拟机器' },
    1_000_000,
  );
  const session = simulation.sessions[0];
  const episode = session.episodes[0];
  const source = episode.segments.find(
    (segment) => segment.source === 'human' && segment.expertValid,
  );
  assert.ok(source && source.annotations.length, 'seed provides an existing expert annotation');
  const callbacks = {
    onSeek() {
      assert.fail('SSR must not seek or mutate parent playback');
    },
    onSave() {
      assert.fail('SSR must not apply an annotation');
    },
    onRemove() {
      assert.fail('SSR must not delete an annotation');
    },
    onClose() {
      assert.fail('SSR must not close the parent editor');
    },
  };
  const editor = (segment = source, sourceEpisode = episode, props = {}) =>
    render(
      React.createElement(Editor, {
        session,
        episode: sourceEpisode,
        segment,
        cursorFrame: segment.startFrame + 1,
        ...callbacks,
        ...props,
      }),
    );

  await test('sealed source segments expose accessible structured labels with explicit bounded frame inputs', () => {
    const before = JSON.stringify({ session, episode, source });
    freeze(session);
    const html = editor();
    for (const label of [
      '标注起始帧',
      '标注终止帧（不含）',
      '任务结果',
      '问题标签',
      '纠正意图',
      '标注备注',
    ])
      assert.ok(html.includes(label), label);
    assert.match(html, /type="number"/);
    assert.ok(html.includes(`min="${source.startFrame}"`));
    assert.ok(html.includes(`max="${source.endFrame}"`));
    assert.ok(html.includes(`value="${source.startFrame}"`));
    assert.ok(html.includes(`value="${source.endFrame}"`));
    assert.match(html, /语义打标独立记录，不改变/);
    assert.match(html, /expert_label_valid = true/);
    assert.match(html, /保留\/排除仍按原片段审核/);
    noExternalMedia(html);
    assert.equal(JSON.stringify({ session, episode, source }), before);
  });

  await test('an empty new label cannot save, while existing labels keep distinct edit and delete controls', () => {
    const extra = {
      ...source.annotations[0],
      id: 'mock-annotation-second',
      startFrame: source.startFrame + 20,
      endFrame: source.startFrame + 40,
      outcome: 'failure',
      issues: ['grasp_failure'],
      intent: 'regrasp',
      note: '模拟：抓取失败后重新夹取。',
    };
    const labelled = freeze({ ...source, annotations: [...source.annotations, extra] });
    const html = editor(labelled);
    const controls = buttonMarkup(html);
    const save = controls.find((button) => button.text.includes('保存标注'));
    assert.ok(save, 'the label action is explicit');
    assert.match(save.html, /disabled/);
    assert.ok(controls.filter((button) => /编辑/.test(button.text)).length >= 2);
    assert.ok(controls.filter((button) => /删除/.test(button.text)).length >= 2);
    assert.match(html, /模拟：抓取失败后重新夹取。/);
    noExternalMedia(html);
  });

  await test('opening an existing annotation populates its own interval and fields, while unknown IDs start a new label', () => {
    const annotation = freeze({
      ...source.annotations[0],
      id: 'mock-edit-target',
      startFrame: source.startFrame + 20,
      endFrame: source.startFrame + 40,
      outcome: 'failure',
      issues: ['grasp_failure'],
      intent: 'regrasp',
      note: '模拟：需调整夹爪后再次抓取。',
    });
    const labelled = freeze({ ...source, annotations: [...source.annotations, annotation] });
    const before = JSON.stringify(labelled);
    const existing = editor(labelled, episode, { initialAnnotationId: annotation.id });
    assert.match(existing, /编辑现有标注/);
    assert.ok(existing.includes(`value="${annotation.startFrame}"`));
    assert.ok(existing.includes(`value="${annotation.endFrame}"`));
    assert.match(existing, /模拟：需调整夹爪后再次抓取。/);
    const save = buttonMarkup(existing).find((button) => /保存修改/.test(button.text));
    assert.ok(save);
    assert.doesNotMatch(save.html, /disabled/);
    const blank = editor(labelled, episode, { initialAnnotationId: 'another-segment-label' });
    assert.match(blank, /新建片段标注/);
    assert.ok(blank.includes(`value="${source.startFrame}"`));
    assert.ok(blank.includes(`value="${source.endFrame}"`));
    const blankSave = buttonMarkup(blank).find((button) => /保存标注/.test(button.text));
    assert.ok(blankSave);
    assert.match(blankSave.html, /disabled/);
    assert.equal(JSON.stringify(labelled), before);
    noExternalMedia(existing);
    noExternalMedia(blank);
  });

  await test('only an integer cursor inside the source interval can set annotation boundaries', () => {
    for (const cursorFrame of [
      source.startFrame - 1,
      source.endFrame,
      source.startFrame + 0.5,
      Number.NaN,
      Number.POSITIVE_INFINITY,
    ]) {
      const html = editor(source, episode, { cursorFrame });
      const boundaries = buttonMarkup(html).filter((button) =>
        /当前帧设为[起终]点/.test(button.text),
      );
      assert.equal(boundaries.length, 2);
      for (const boundary of boundaries) assert.match(boundary.html, /disabled/);
      assert.match(html, /游标不在当前片段内/);
    }
    const inside = editor(source, episode, { cursorFrame: source.endFrame - 1 });
    for (const boundary of buttonMarkup(inside).filter((button) =>
      /当前帧设为[起终]点/.test(button.text),
    ))
      assert.doesNotMatch(boundary.html, /disabled/);
    assert.match(inside, /游标 \+ 1/);
    assert.match(inside, /包含当前帧/);
  });

  await test('live or empty source intervals cannot expose an enabled annotation save action', () => {
    for (const [segment, sourceEpisode] of [
      [source, { ...episode, status: 'recording' }],
      [{ ...source, endFrame: source.startFrame }, episode],
    ]) {
      const html = editor(segment, sourceEpisode);
      for (const save of buttonMarkup(html).filter((button) => button.text.includes('保存标注')))
        assert.match(save.html, /disabled/);
      assert.match(html, /封存|非空|有效|结束/);
      noExternalMedia(html);
    }
  });

  await test('policy and invalid human context can receive descriptive labels without changing expert qualification', () => {
    for (const segment of [
      freeze({
        ...source,
        id: 'policy-context',
        source: 'policy',
        expertValid: false,
        annotations: [],
      }),
      freeze({
        ...source,
        id: 'invalid-human-context',
        source: 'human',
        expertValid: false,
        annotations: [],
      }),
    ]) {
      const before = JSON.stringify(segment);
      const html = editor(segment);
      assert.match(html, /标注|标签/);
      assert.match(html, /专家/);
      assert.doesNotMatch(html, /<(?:input|select)[^>]*(?:name|id)="expertValid"/);
      assert.equal(JSON.stringify(segment), before);
      assert.equal(segment.expertValid, false);
      noExternalMedia(html);
    }
  });

  await test('compact and full summaries show safe human-readable labels and preserve the source annotations', () => {
    const annotations = freeze([
      {
        ...source.annotations[0],
        note: '<img src="external" onerror="example">',
      },
    ]);
    const before = JSON.stringify(annotations);
    const compact = render(React.createElement(Summary, { annotations, compact: true }));
    const full = render(React.createElement(Summary, { annotations }));
    const empty = render(React.createElement(Summary, { annotations: [] }));
    assert.match(empty, /未打标/);
    assert.match(compact, /标注|标签|打标/);
    assert.ok(full.includes(String(annotations[0].startFrame)));
    assert.ok(full.includes(String(annotations[0].endFrame)));
    assert.match(full, /&lt;img/);
    noExternalMedia(compact);
    noExternalMedia(full);
    assert.equal(JSON.stringify(annotations), before);
  });

  await test('the unified mock workspace exposes source replay and a labeling entry without a second outer card', () => {
    const html = render(
      React.createElement(DataReturnMock, {
        deploymentRecordId: 'record-label-preview',
        deploymentName: '标注演示部署',
        machineName: '模拟机器',
      }),
    );
    assert.equal((html.match(/data-testid="data-return-mock"/g) || []).length, 1);
    assert.match(html, /aria-label="控制来源时间线"/);
    assert.match(html, /aria-label="双相机预览"/);
    assert.ok(buttonMarkup(html).some((button) => button.text === '片段打标'));
    assert.match(html, /打标与审核/);
    assert.doesNotMatch(html, /data-testid="data-return-annotation-editor"/);
    noExternalMedia(html);
  });
} finally {
  await server.close();
}
