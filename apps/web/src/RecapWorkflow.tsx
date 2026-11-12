import { useId, useRef, useState } from 'react';
import {
  ArrowRight,
  ChartNoAxesColumnIncreasing,
  Check,
  ChevronDown,
  Copy,
  Tags,
  Target,
} from 'lucide-react';
import { RECAP_PYTORCH_ENTRYPOINT, recapStageSupervision } from './recap';
import type { TrainingStage, TrainingStageSpec } from './types';
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
  onSelect,
}: {
  stages: TrainingStageSpec[];
  selected?: TrainingStage;
  disabled: boolean;
  onSelect: (stage: TrainingStage) => void;
}) {
  const headingId = useId();
  const annotationId = useId();
  const [annotationOpen, setAnnotationOpen] = useState(false);
  const [copyStatus, setCopyStatus] = useState('');
  const commandField = useRef<HTMLTextAreaElement>(null);
  const acpButton = useRef<HTMLButtonElement>(null);
  const valueStage = stages.find(({ id }) => id === 'value');
  const acpStage = stages.find(({ id }) => id === 'offline_rl');
  const kai0Stage = stages.find(({ id }) => String(id) === 'stage_progress');
  const valueUsesPytorch = valueStage?.default_entrypoint === RECAP_PYTORCH_ENTRYPOINT;
  const acpUsesPytorch = acpStage?.default_entrypoint === RECAP_PYTORCH_ENTRYPOINT;

  const choose = (stage: TrainingStage) => {
    setAnnotationOpen(false);
    if (selected !== stage) onSelect(stage);
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
        <p>Value 和 ACP 分别提交训练作业。请选择本次作业。</p>
      </header>
      <div className="recap-workflow-steps">
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
          aria-expanded={annotationOpen}
          aria-controls={annotationId}
          onClick={() => {
            setAnnotationOpen(!annotationOpen);
            setCopyStatus('');
          }}
        >
          <span className="recap-workflow-icon">
            <Tags size={17} aria-hidden="true" />
          </span>
          <span className="recap-workflow-card-content">
            <span className="recap-workflow-caption">中间步骤 · 在训练机执行</span>
            <strong>优势值标注</strong>
            <span>
              使用 Value 检查点，生成含 <code>{recapStageSupervision.offline_rl}</code> 的数据
            </span>
          </span>
          <ChevronDown size={16} className="recap-workflow-chevron" aria-hidden="true" />
        </button>
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
              训练阶段 2{acpStage && (acpUsesPytorch ? ' · PyTorch' : ' · JAX')}
            </span>
            <strong>ACP 训练</strong>
            <span>使用标注数据和初始策略权重，产出 ACP 策略</span>
          </span>
          {selected === 'offline_rl' && (
            <Check size={16} className="recap-workflow-check" aria-hidden="true" />
          )}
        </button>
      </div>
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
      {kai0Stage && (
        <div className="recap-workflow-kai0">
          <button
            type="button"
            className="button outline small"
            aria-pressed={selected === kai0Stage.id}
            disabled={disabled}
            onClick={() => choose(kai0Stage.id)}
          >
            {selected === kai0Stage.id && <Check size={14} aria-hidden="true" />}
            使用 Kai0 进度标注
          </button>
          <p>
            备选路线。使用单个含 <code>{recapStageSupervision.stage_progress}</code>{' '}
            的数据集，训练并标注后将派生数据用于 ACP。
          </p>
        </div>
      )}
    </section>
  );
}
