import { useId, useRef, useState } from 'react';
import {
  ArrowRight,
  ChartNoAxesColumnIncreasing,
  Check,
  ChevronDown,
  Copy,
  Layers3,
  Tags,
  Target,
} from 'lucide-react';
import {
  isStageAdvantageStage,
  RECAP_PYTORCH_ENTRYPOINT,
  recapStageSupervision,
  STAGE_ADVANTAGE_MODE,
} from './recap';
import type { TrainingStage, TrainingStageSpec } from './types';
import type { RecapMode } from './recapPhaseDraft';
import './recap-workflow.css';

const annotationCommand = [
  'uv run python scripts/acp/lerobot_value_infer_pytorch.py',
  '  --config-file /path/to/run/training_config.py',
  '  --checkpoint-dir /path/to/value/checkpoint_step',
  '  --dataset-root /path/to/lerobot_with_value_target',
  '  --output-dir /path/to/derived_dataset',
].join(' \\\n');

export function RecapWorkflow({
  stages,
  selected,
  disabled,
  mode = isStageAdvantageStage(selected) ? 'stage_advantage' : 'recap',
  onSelect,
}: {
  stages: TrainingStageSpec[];
  selected?: TrainingStage;
  disabled: boolean;
  mode?: RecapMode;
  onSelect: (stage: TrainingStage, mode: RecapMode) => void;
}) {
  const headingId = useId();
  const annotationId = useId();
  const [annotationOpen, setAnnotationOpen] = useState(false);
  const [copyStatus, setCopyStatus] = useState('');
  const commandField = useRef<HTMLTextAreaElement>(null);
  const acpButton = useRef<HTMLButtonElement>(null);
  const valueStage = stages.find(({ id }) => id === 'value');
  const acpStage = stages.find(({ id }) => id === 'offline_rl');
  const kai0Stage = stages.find(({ id }) => id === 'stage_progress');
  const stageAnnotation = stages.find(({ id }) => id === 'stage_annotation');
  const valueAnnotation = stages.find(({ id }) => id === 'value_annotation');
  const kai0Mode = !!kai0Stage && mode === 'stage_advantage';
  const valueUsesPytorch =
    valueStage?.backend === 'pytorch' ||
    valueStage?.default_entrypoint === RECAP_PYTORCH_ENTRYPOINT;

  const choose = (stage: TrainingStage) => {
    setAnnotationOpen(false);
    if (selected !== stage) onSelect(stage, mode);
  };
  const changeMode = (next: RecapMode) => {
    if (next === mode) return;
    setAnnotationOpen(false);
    onSelect(
      selected === 'offline_rl'
        ? 'offline_rl'
        : next === 'stage_advantage'
          ? 'stage_progress'
          : 'value',
      next,
    );
  };
  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(annotationCommand);
      setCopyStatus('已复制命令');
    } catch {
      commandField.current?.focus();
      commandField.current?.select();
      setCopyStatus('命令已选中，可按复制快捷键。');
    }
  };

  return (
    <section className="recap-workflow" aria-labelledby={headingId}>
      <header className="recap-workflow-heading">
        <h3 id={headingId}>ReCap 工作流</h3>
      </header>
      {kai0Stage && (
        <div className="recap-workflow-modes" role="group" aria-label="训练模式">
          <button
            type="button"
            className="recap-workflow-mode"
            aria-label="ReCap 模式"
            aria-pressed={!kai0Mode}
            disabled={disabled || !valueStage}
            onClick={() => changeMode('recap')}
          >
            <ChartNoAxesColumnIncreasing size={17} aria-hidden="true" />
            <span>
              <strong>ReCap 模式</strong>
              <span>
                回报价值 · <code>{recapStageSupervision.value}</code>
              </span>
            </span>
            {!kai0Mode && <Check className="recap-mode-check" size={15} aria-hidden="true" />}
          </button>
          <button
            type="button"
            className="recap-workflow-mode"
            aria-label={STAGE_ADVANTAGE_MODE}
            aria-pressed={kai0Mode}
            disabled={disabled}
            onClick={() => changeMode('stage_advantage')}
          >
            <Layers3 size={17} aria-hidden="true" />
            <span>
              <strong>{STAGE_ADVANTAGE_MODE}</strong>
              <span>使用指定数据列中的逐帧阶段进度</span>
            </span>
            {kai0Mode && <Check className="recap-mode-check" size={15} aria-hidden="true" />}
          </button>
        </div>
      )}
      <div className={`recap-workflow-steps${kai0Mode && !stageAnnotation ? ' kai0' : ''}`}>
        {kai0Mode ? (
          <>
            <button
              type="button"
              className="recap-workflow-card"
              aria-pressed={selected === 'stage_progress'}
              disabled={disabled}
              onClick={() => choose('stage_progress')}
            >
              <span className="recap-workflow-icon">
                <Layers3 size={17} aria-hidden="true" />
              </span>
              <span className="recap-workflow-card-content">
                <span className="recap-workflow-caption">训练阶段 1 · PyTorch</span>
                <strong>{stageAnnotation ? '阶段进度训练' : '阶段进度训练与优势标注'}</strong>
                {stageAnnotation ? (
                  <span>用指定的阶段进度列训练估计器，产出模型检查点</span>
                ) : (
                  <span>
                    用指定的阶段进度列产出含 <code>{recapStageSupervision.offline_rl}</code>{' '}
                    的派生数据
                  </span>
                )}
              </span>
              {selected === 'stage_progress' && (
                <Check size={16} className="recap-workflow-check" aria-hidden="true" />
              )}
            </button>
            {stageAnnotation && (
              <>
                <ArrowRight size={16} className="recap-workflow-arrow" aria-hidden="true" />
                <button
                  type="button"
                  className="recap-workflow-card"
                  aria-pressed={selected === 'stage_annotation'}
                  disabled={disabled}
                  onClick={() => choose('stage_annotation')}
                >
                  <span className="recap-workflow-icon">
                    <Tags size={17} aria-hidden="true" />
                  </span>
                  <span className="recap-workflow-card-content">
                    <span className="recap-workflow-caption">标注阶段 2 · PyTorch</span>
                    <strong>优势标注</strong>
                    <span>
                      选择阶段进度检查点和待标注数据集，生成含{' '}
                      <code>{recapStageSupervision.offline_rl}</code> 的数据
                    </span>
                  </span>
                  {selected === 'stage_annotation' && (
                    <Check size={16} className="recap-workflow-check" aria-hidden="true" />
                  )}
                </button>
              </>
            )}
          </>
        ) : (
          <>
            <button
              type="button"
              className="recap-workflow-card"
              aria-pressed={selected === 'value'}
              disabled={disabled || !valueStage}
              onClick={() => choose('value')}
            >
              <span className="recap-workflow-icon">
                <ChartNoAxesColumnIncreasing size={17} aria-hidden="true" />
              </span>
              <span className="recap-workflow-card-content">
                <span className="recap-workflow-caption">
                  训练阶段 1{valueStage && (valueUsesPytorch ? ' · PyTorch' : ' · JAX')}
                </span>
                <strong>Value 训练</strong>
                <span>
                  使用 <code>{recapStageSupervision.value}</code>，产出 Value 检查点
                </span>
              </span>
              {selected === 'value' && (
                <Check size={16} className="recap-workflow-check" aria-hidden="true" />
              )}
            </button>
            <ArrowRight size={16} className="recap-workflow-arrow" aria-hidden="true" />
            <button
              type="button"
              className="recap-workflow-card"
              aria-pressed={valueAnnotation ? selected === 'value_annotation' : undefined}
              aria-expanded={!valueAnnotation ? annotationOpen : undefined}
              aria-controls={!valueAnnotation ? annotationId : undefined}
              disabled={disabled}
              onClick={() => {
                if (valueAnnotation) choose('value_annotation');
                else {
                  setAnnotationOpen(!annotationOpen);
                  setCopyStatus('');
                }
              }}
            >
              <span className="recap-workflow-icon">
                <Tags size={17} aria-hidden="true" />
              </span>
              <span className="recap-workflow-card-content">
                <span className="recap-workflow-caption">
                  {valueAnnotation ? '标注阶段 2 · PyTorch' : '中间步骤 · 在训练机执行'}
                </span>
                <strong>优势值标注</strong>
                <span>
                  使用 Value 检查点，生成含 <code>{recapStageSupervision.offline_rl}</code> 的数据
                </span>
              </span>
              {valueAnnotation ? (
                selected === 'value_annotation' && (
                  <Check size={16} className="recap-workflow-check" aria-hidden="true" />
                )
              ) : (
                <ChevronDown size={16} className="recap-workflow-chevron" aria-hidden="true" />
              )}
            </button>
          </>
        )}
        <ArrowRight size={16} className="recap-workflow-arrow" aria-hidden="true" />
        <button
          type="button"
          className="recap-workflow-card"
          ref={acpButton}
          aria-pressed={selected === 'offline_rl'}
          disabled={disabled || !acpStage}
          onClick={() => choose('offline_rl')}
        >
          <span className="recap-workflow-icon">
            <Target size={17} aria-hidden="true" />
          </span>
          <span className="recap-workflow-card-content">
            <span className="recap-workflow-caption">
              训练阶段 {(kai0Mode ? stageAnnotation : valueAnnotation) ? 3 : 2}
              {acpStage && ' · JAX'}
            </span>
            <strong>ACP 训练</strong>
            <span>
              使用含 <code>{recapStageSupervision.offline_rl}</code> 的数据和 JAX / Orbax
              初始策略权重
            </span>
          </span>
          {selected === 'offline_rl' && (
            <Check size={16} className="recap-workflow-check" aria-hidden="true" />
          )}
        </button>
      </div>
      {!kai0Mode && !valueAnnotation && (
        <div className="recap-workflow-annotation" id={annotationId} hidden={!annotationOpen}>
          {valueUsesPytorch ? (
            <>
              <div className="recap-workflow-annotation-heading">
                <p>
                  在训练机的 OpenPI-UMI 目录运行。将路径替换为 Value 作业导出的{' '}
                  <code>training_config.py</code>、Value 检查点、含{' '}
                  <code>{recapStageSupervision.value}</code> 的待标注数据和新输出目录。
                </p>
                <button
                  type="button"
                  className="button outline small"
                  onClick={() => void copyCommand()}
                >
                  <Copy size={14} aria-hidden="true" />
                  复制命令
                </button>
              </div>
              <textarea
                ref={commandField}
                aria-label="优势值标注命令"
                value={annotationCommand}
                rows={5}
                readOnly
                spellCheck={false}
              />
              <span className="recap-workflow-copy-status" role="status">
                {copyStatus}
              </span>
            </>
          ) : (
            <p>
              在训练机复制含 <code>{recapStageSupervision.value}</code> 的数据集。使用与 Value
              检查点匹配的 OpenPI-UMI 配方，通过 <code>scripts/lerobot_value_infer.py</code>{' '}
              给副本写入 <code>{recapStageSupervision.offline_rl}</code>。
            </p>
          )}
          <div className="recap-workflow-annotation-footer">
            <p>
              {valueUsesPytorch && (
                <>
                  多数据集配方可用 <code>--dataset-index</code> 选择对应配置。
                </>
              )}
              标注完成后，在「数据与模型」登记派生数据，再为 ACP 选择该数据。
            </p>
            {acpStage && (
              <button
                type="button"
                className="button outline small"
                disabled={disabled}
                onClick={() => {
                  choose('offline_rl');
                  acpButton.current?.focus();
                }}
              >
                已有标注数据，配置 ACP
                <ArrowRight size={14} aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
