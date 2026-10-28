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
try {
  const { trainingInferenceConfig, sourceDeploymentBlocker } = await server.ssrLoadModule(
    '/src/policyDeployment.ts',
  );
  const { currentRecipeEntries, recipeSuggestionScope } = await server.ssrLoadModule(
    '/src/recipeSuggestions.ts',
  );
  const { RecipeAutocomplete } = await server.ssrLoadModule('/src/RecipeAutocomplete.tsx');
  const sha256 = 'a'.repeat(64);
  const frozen = {
    recipe: 'auto_pi05_confirmed',
    confirmed_config_sha256: sha256,
    auto_config: { model_type: 'pi05', action_dim: 14, action_horizon: 32, finetune_mode: 'full' },
    automatic_preflight: {
      plan: {
        datasets: [{ path: '/data/cloth' }],
        generated_config: {
          schema_version: 1,
          language: 'python',
          filename: 'training_config.py',
          source: 'confirmed config source',
          sha256,
        },
      },
    },
  };
  const run = {
    id: 'run-a',
    template_id: 'openpi-sft',
    state: 'SUCCEEDED',
    recipe: 'base_recipe',
    code_expectation: frozen,
  };
  const node = { id: 'target-node', status: 'ONLINE' };
  const profile = {
    id: 'target-profile',
    node_id: node.id,
    code_dir: '/target/openpi',
    python: '/target/python',
    recipe: 'old_registered',
    metadata: { validated_recipes: ['old_registered'] },
  };
  const response = {
    status: 'READY',
    node_id: node.id,
    profile_id: profile.id,
    prefix: 'pi',
    entries: [
      { name: 'pi05_code', source: 'code', validated: false },
      { name: 'pi0_both', source: 'both', validated: true },
      { name: 'pi_legacy_registration', source: 'registered', validated: true },
    ],
  };

  await test('training inference uses the confirmed frozen config and model parameters', () => {
    assert.deepEqual(trainingInferenceConfig(run), {
      recipe: frozen.recipe,
      sha256,
      autoConfig: frozen.auto_config,
    });
    assert.equal(sourceDeploymentBlocker(run), undefined);
    assert.equal(
      trainingInferenceConfig({ ...run, recipe: 'different_mutable_recipe' }).recipe,
      frozen.recipe,
    );
    assert.equal(
      trainingInferenceConfig({
        ...run,
        confirmed_config_sha256: sha256,
        code_expectation: { ...frozen, confirmed_config_sha256: undefined },
      }),
      undefined,
    );
    assert.equal(
      trainingInferenceConfig({
        ...run,
        code_expectation: {
          ...frozen,
          recipe: undefined,
          automatic_preflight: {
            plan: { ...frozen.automatic_preflight.plan, recipe: frozen.recipe },
          },
        },
      }).recipe,
      frozen.recipe,
    );
  });

  await test('legacy or mismatched run configs block automatic inference instead of falling back to a base recipe', () => {
    for (const code_expectation of [
      undefined,
      { recipe: 'base_recipe' },
      { ...frozen, recipe: undefined },
      { ...frozen, confirmed_config_sha256: 'b'.repeat(64) },
      {
        ...frozen,
        automatic_preflight: {
          plan: { ...frozen.automatic_preflight.plan, recipe: 'mismatched_recipe' },
        },
      },
      { ...frozen, automatic_preflight: { plan: { generated_config: { source: '', sha256 } } } },
      {
        ...frozen,
        automatic_preflight: {
          plan: { generated_config: { source: 'config', sha256: 'not-a-digest' } },
        },
      },
    ]) {
      const legacy = { ...run, code_expectation };
      assert.equal(trainingInferenceConfig(legacy), undefined);
      assert.match(sourceDeploymentBlocker(legacy), /缺少已确认的生成配置/);
    }
    assert.match(sourceDeploymentBlocker({ ...run, state: 'RUNNING' }), /训练尚未结束/);
  });

  await test('manual inference accepts only current target code discoveries, including exact typed matches', () => {
    assert.deepEqual(
      currentRecipeEntries(response, node, profile, 'pi', true).map((entry) => entry.name),
      ['pi05_code', 'pi0_both'],
    );
    assert.equal(currentRecipeEntries(response, node, profile, 'pi', false).length, 3);
    for (const [result, target, env, value] of [
      [response, { id: 'other-node' }, profile, 'pi'],
      [response, node, { ...profile, id: 'other-profile' }, 'pi'],
      [response, node, { ...profile, node_id: 'other-node' }, 'pi'],
      [response, node, profile, 'pi05_code'],
      [{ ...response, status: 'PENDING' }, node, profile, 'pi'],
    ])
      assert.equal(currentRecipeEntries(result, target, env, value, true), undefined);
    const exact = { ...response, prefix: 'pi05_code' };
    assert.equal(
      currentRecipeEntries(exact, node, profile, exact.prefix, true).find(
        (entry) => entry.name === exact.prefix,
      )?.name,
      exact.prefix,
    );
    assert.equal(
      currentRecipeEntries({ ...response, prefix: 'unknown' }, node, profile, 'unknown', true).find(
        (entry) => entry.name === 'unknown',
      ),
      undefined,
    );
  });

  await test('a target/profile checkout or endpoint switch invalidates old suggestion scope', () => {
    const scope = recipeSuggestionScope(node, profile, 'inference-config-suggestions');
    for (const changed of [
      { ...profile, id: 'other-profile' },
      { ...profile, code_dir: '/new/checkout' },
      { ...profile, python: '/new/python' },
      { ...profile, metadata: { validated_recipes: ['another'] } },
    ])
      assert.notEqual(recipeSuggestionScope(node, changed, 'inference-config-suggestions'), scope);
    assert.notEqual(
      recipeSuggestionScope({ ...node, id: 'other-node' }, profile, 'inference-config-suggestions'),
      scope,
    );
    assert.notEqual(recipeSuggestionScope(node, profile), scope);
    assert.equal(
      recipeSuggestionScope({ ...node }, { ...profile }, 'inference-config-suggestions'),
      scope,
    );
  });

  await test('inference autocomplete uses inference language while training defaults remain unchanged', () => {
    const inference = renderToStaticMarkup(
      React.createElement(RecipeAutocomplete, {
        value: '',
        onChange() {},
        node,
        profile,
        context: 'inference',
        codeOnly: true,
        namespace: 'inference-config-suggestions',
      }),
    );
    assert.match(inference, /推理配置名称/);
    assert.match(inference, /选择目标运行环境代码中的已有配置/);
    assert.doesNotMatch(inference, /训练配方|预检时/);
    const training = renderToStaticMarkup(
      React.createElement(RecipeAutocomplete, { value: '', onChange() {} }),
    );
    assert.match(training, /训练配方/);
    assert.match(training, /预检时自动验证/);
  });
} finally {
  await server.close();
}
