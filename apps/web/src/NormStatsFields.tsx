import { RecipeAutocomplete } from './RecipeAutocomplete';
import { Field } from './components';
import { DEFAULT_AUTO_CONFIG, type DatasetActionHorizonInference } from './trainingConfiguration';
import type { AutoTrainingConfig, Node, Profile } from './types';

export function NormStatsFields({
  node,
  profile,
  recipe,
  onChange,
  advanced = false,
  onAdvancedChange,
  autoConfig,
  onAutoConfigChange,
  horizonInference,
  horizonMode = 'automatic',
  onHorizonModeChange,
  showAutomaticFields = true,
}: {
  node?: Node;
  profile?: Profile;
  recipe: string;
  onChange: (recipe: string) => void;
  advanced?: boolean;
  onAdvancedChange?: (advanced: boolean) => void;
  autoConfig?: AutoTrainingConfig;
  onAutoConfigChange?: (config: AutoTrainingConfig) => void;
  horizonInference?: DatasetActionHorizonInference;
  horizonMode?: 'automatic' | 'manual';
  onHorizonModeChange?: (mode: 'automatic' | 'manual') => void;
  showAutomaticFields?: boolean;
}) {
  const config = autoConfig || DEFAULT_AUTO_CONFIG;
  return (
    <section aria-label="norm_stats 计算配置">
      <div className="inline-note">
        按每组配置完整遍历数据集，计算 state / actions
        的归一化统计。多个数据集进入批量计算模式，逐个独立生成 norm_stats；任务使用
        CPU，完成后自动登记各自的归一化参数资产。
      </div>
      {!advanced && showAutomaticFields && (
        <>
          <p className="field-hint">
            平台根据动作配置和每个数据集的字段组合构造统计 config，无需填写代码配方。
          </p>
          <div className="form-grid">
            <Field
              label="动作预测长度"
              required
              hint="用于核对已存储动作块的长度。每行统计一个动作向量，动作块取第一步。"
            >
              <input
                required
                type="number"
                min={1}
                max={1000}
                step={1}
                value={Number.isFinite(config.action_horizon) ? config.action_horizon : ''}
                onChange={(event) => {
                  onHorizonModeChange?.('manual');
                  onAutoConfigChange?.({
                    ...config,
                    action_horizon: event.target.value === '' ? NaN : Number(event.target.value),
                    finetune_mode: 'full',
                  });
                }}
              />
              <div className="field-hint">
                {horizonMode === 'manual' ? '手动设置 · 自动识别不会覆盖' : '自动默认值'}
                {horizonInference?.sources
                  .filter((source) => source.shape)
                  .map((source, index) => (
                    <div key={index}>
                      {source.assetName} · {source.actionKey} [{source.shape!.join(', ')}]
                    </div>
                  ))}
              </div>
              {horizonMode === 'manual' && (
                <button
                  type="button"
                  className="button outline small"
                  onClick={() => {
                    onHorizonModeChange?.('automatic');
                    onAutoConfigChange?.({
                      ...config,
                      action_horizon:
                        horizonInference?.recommendedHorizon ?? DEFAULT_AUTO_CONFIG.action_horizon,
                      finetune_mode: 'full',
                    });
                  }}
                >
                  {horizonInference?.recommendedHorizon !== undefined
                    ? `使用数据集长度 ${horizonInference.recommendedHorizon}`
                    : '恢复默认值 50'}
                </button>
              )}
              {horizonInference?.errors.map((message, index) => (
                <div className="inline-note red" role="alert" key={index}>
                  {message}
                </div>
              ))}
            </Field>
            <Field
              label="模型动作维度"
              required
              hint="用于核对状态与动作的填充兼容性，不需要加载模型权重。"
            >
              <input
                required
                type="number"
                min={1}
                max={4096}
                step={1}
                value={Number.isFinite(config.action_dim) ? config.action_dim : ''}
                onChange={(event) =>
                  onAutoConfigChange?.({
                    ...config,
                    action_dim: event.target.value === '' ? NaN : Number(event.target.value),
                    finetune_mode: 'full',
                  })
                }
              />
            </Field>
          </div>
        </>
      )}
      <details className="training-advanced-settings" open={advanced || undefined}>
        <summary>高级设置：代码配方</summary>
        <label className="training-mode-choice">
          <input
            type="checkbox"
            checked={advanced}
            onChange={(event) => onAdvancedChange?.(event.target.checked)}
          />
          <span>使用代码配方构造统计配置</span>
        </label>
        {advanced && (
          <RecipeAutocomplete
            node={node}
            profile={profile}
            value={recipe}
            onChange={onChange}
            label="统计使用的代码配方"
            hint="配方提供动作预测长度等配置。每个数据集可沿用单数据集配方、选择多数据集配方中的数据项，或独立配置处理器。"
          />
        )}
      </details>
    </section>
  );
}
