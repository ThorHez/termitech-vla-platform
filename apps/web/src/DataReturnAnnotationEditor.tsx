import { useId, useState, type FormEvent } from 'react';
import {
  Check,
  CheckCircle2,
  ChevronRight,
  Crosshair,
  Pencil,
  Plus,
  Save,
  Tag,
  Trash2,
  X,
} from 'lucide-react';
import {
  annotationValidationError,
  type DataReturnAnnotation,
  type DataReturnAnnotationInput,
  type DataReturnEpisode,
  type DataReturnSegment,
  type DataReturnSession,
} from './dataReturnSimulation';
import './data-return-annotations.css';

type Outcome = DataReturnAnnotation['outcome'];
type Issue = DataReturnAnnotation['issues'][number];
type Intent = DataReturnAnnotation['intent'];
type AnnotationSummaryItem = Readonly<Omit<DataReturnAnnotation, 'issues'>> & {
  readonly issues: readonly Issue[];
};

const outcomes: Record<Outcome, string> = { unknown: '待判定', success: '成功', failure: '失败' };
const issues: Record<Issue, string> = {
  grasp_failure: '抓取失败',
  pose_error: '位置偏差',
  collision_risk: '碰撞风险',
  occlusion: '视觉遮挡',
  unstable_motion: '动作抖动',
  other: '其他',
};
const intents: Record<Intent, string> = {
  none: '未设置',
  align_pose: '对齐位姿',
  adjust_gripper: '调整夹爪',
  regrasp: '重新抓取',
  recover_trajectory: '恢复轨迹',
  complete_task: '任务完成',
};
const rangeLabel = (annotation: Pick<DataReturnAnnotation, 'startFrame' | 'endFrame'>) =>
  `${annotation.startFrame}–${annotation.endFrame} 帧`;

/** Both editable annotations and frozen version snapshots can use this presentation. */
export function DataReturnAnnotationSummary({
  annotations,
  compact = false,
}: {
  annotations: readonly AnnotationSummaryItem[];
  compact?: boolean;
}) {
  if (!annotations.length) return <span className="dra-summary-empty">未打标</span>;
  if (compact) {
    const uniqueIssues = [...new Set(annotations.flatMap((annotation) => annotation.issues))];
    const resolvedOutcomes = (['success', 'failure'] as const).filter((outcome) =>
      annotations.some((annotation) => annotation.outcome === outcome),
    );
    const uniqueIntents = [...new Set(annotations.map((annotation) => annotation.intent))].filter(
      (intent) => intent !== 'none',
    );
    const labels = [
      ...resolvedOutcomes.map((outcome) => ({ label: outcomes[outcome], tone: outcome })),
      ...uniqueIssues.map((issue) => ({ label: issues[issue], tone: 'issue' })),
      ...uniqueIntents.map((intent) => ({ label: intents[intent], tone: 'intent' })),
    ];
    return (
      <span className="dra-summary compact" aria-label={`打标摘要，共 ${annotations.length} 条`}>
        <span className="dra-summary-count">
          <Tag size={10} />
          {annotations.length} 条打标
        </span>
        <span className="dra-summary-interval" title={annotations.map(rangeLabel).join('；')}>
          {rangeLabel(annotations[0])}
          {annotations.length > 1 ? ' 等' : ''}
        </span>
        {labels.slice(0, 3).map((label) => (
          <span className={`dra-chip ${label.tone}`} key={`${label.tone}-${label.label}`}>
            {label.label}
          </span>
        ))}
        {labels.length > 3 && (
          <span
            className="dra-summary-more"
            title={labels
              .slice(3)
              .map((label) => label.label)
              .join('、')}
          >
            +{labels.length - 3}
          </span>
        )}
      </span>
    );
  }
  return (
    <div
      className="dra-summary full"
      aria-label={`打标摘要，共 ${annotations.length} 条`}
      role="list"
    >
      {annotations.map((annotation) => (
        <div className="dra-summary-item" key={annotation.id} role="listitem">
          <span className="dra-summary-range">
            {rangeLabel(annotation)}
            <small>终点不含</small>
          </span>
          <AnnotationLabels annotation={annotation} />
          {annotation.note && <p>{annotation.note}</p>}
        </div>
      ))}
    </div>
  );
}

function AnnotationLabels({ annotation }: { annotation: AnnotationSummaryItem }) {
  return (
    <span className="dra-labels">
      <span className={`dra-chip ${annotation.outcome}`}>{outcomes[annotation.outcome]}</span>
      {annotation.issues.map((issue) => (
        <span className="dra-chip issue" key={issue}>
          {issues[issue]}
        </span>
      ))}
      {annotation.intent !== 'none' && (
        <span className="dra-chip intent">{intents[annotation.intent]}</span>
      )}
    </span>
  );
}

export interface DataReturnAnnotationEditorProps {
  session: DataReturnSession;
  episode: DataReturnEpisode;
  segment: DataReturnSegment;
  cursorFrame: number;
  initialAnnotationId?: string;
  onSeek(frame: number): void;
  onSave(annotation: DataReturnAnnotationInput, annotationId?: string): void;
  onRemove(annotationId: string): void;
  onClose(): void;
}

export default function DataReturnAnnotationEditor(props: DataReturnAnnotationEditorProps) {
  return (
    <AnnotationEditorWorkspace
      key={`${props.session.id}:${props.episode.id}:${props.segment.id}:${props.initialAnnotationId || 'new'}`}
      {...props}
    />
  );
}

function AnnotationEditorWorkspace({
  session,
  episode,
  segment,
  cursorFrame,
  initialAnnotationId,
  onSeek,
  onSave,
  onRemove,
  onClose,
}: DataReturnAnnotationEditorProps) {
  const id = useId();
  const initial = segment.annotations.find((annotation) => annotation.id === initialAnnotationId);
  const [editingId, setEditingId] = useState<string | undefined>(initial?.id);
  const [startFrame, setStartFrame] = useState(String(initial?.startFrame ?? segment.startFrame));
  const [endFrame, setEndFrame] = useState(String(initial?.endFrame ?? segment.endFrame));
  const [outcome, setOutcome] = useState<Outcome>(initial?.outcome ?? 'unknown');
  const [selectedIssues, setSelectedIssues] = useState<Issue[]>(initial ? [...initial.issues] : []);
  const [intent, setIntent] = useState<Intent>(initial?.intent ?? 'none');
  const [note, setNote] = useState(initial?.note ?? '');
  const [dirty, setDirty] = useState(false);
  const [feedback, setFeedback] = useState('');
  const sealed = episode.status === 'sealed';
  const editable = sealed && segment.endFrame > segment.startFrame;
  const cursor = cursorFrame;
  const cursorWithinSegment =
    Number.isFinite(cursor) &&
    Number.isInteger(cursor) &&
    cursor >= segment.startFrame &&
    cursor < segment.endFrame;
  const input: DataReturnAnnotationInput = {
    startFrame: startFrame.trim() ? Number(startFrame) : Number.NaN,
    endFrame: endFrame.trim() ? Number(endFrame) : Number.NaN,
    outcome,
    issues: selectedIssues,
    intent,
    note,
  };
  const validationError = annotationValidationError(segment, input);
  const boundaryError =
    !Number.isInteger(input.startFrame) ||
    !Number.isInteger(input.endFrame) ||
    input.startFrame < segment.startFrame ||
    input.endFrame > segment.endFrame ||
    input.startFrame >= input.endFrame;
  const resetDraft = () => {
    setEditingId(undefined);
    setStartFrame(String(segment.startFrame));
    setEndFrame(String(segment.endFrame));
    setOutcome('unknown');
    setSelectedIssues([]);
    setIntent('none');
    setNote('');
    setDirty(false);
  };
  const changed = () => {
    setDirty(true);
    setFeedback('');
  };
  const save = (event: FormEvent) => {
    event.preventDefault();
    setDirty(true);
    if (!editable || validationError) return;
    onSave({ ...input, issues: [...input.issues], note: input.note.trim() }, editingId);
    setFeedback(editingId ? '标注已更新。' : '标注已保存，可继续添加其他范围的标注。');
    resetDraft();
  };
  const edit = (annotation: DataReturnAnnotation) => {
    setEditingId(annotation.id);
    setStartFrame(String(annotation.startFrame));
    setEndFrame(String(annotation.endFrame));
    setOutcome(annotation.outcome);
    setSelectedIssues([...annotation.issues]);
    setIntent(annotation.intent);
    setNote(annotation.note);
    setDirty(false);
    setFeedback('');
    onSeek(annotation.startFrame);
  };

  return (
    <section
      className="dr-annotation-editor"
      aria-label="片段打标编辑器"
      data-testid="data-return-annotation-editor"
    >
      <header className="dra-header">
        <div>
          <span className="dra-heading-icon">
            <Tag size={15} />
          </span>
          <h3>片段打标</h3>
        </div>
        <button
          className="dra-icon-button"
          type="button"
          aria-label="关闭片段打标"
          onClick={onClose}
        >
          <X size={15} />
        </button>
      </header>
      <div className="dra-context">
        <div className="dra-context-path">
          <span title={session.name}>{session.name}</span>
          <ChevronRight size={10} />
          <span title={episode.name}>{episode.name}</span>
        </div>
        <div className="dra-source-row">
          <span className={`dra-source ${segment.source}`}>
            {segment.source === 'human' ? '人工来源' : '策略上下文'}
          </span>
          <span>
            {segment.startFrame}–{segment.endFrame} 帧 <small>终点不含</small>
          </span>
        </div>
        <p>
          语义打标独立记录，不改变 <code>expert_label_valid = {String(segment.expertValid)}</code>。
        </p>
        <p>打标区间用于描述片段，保留/排除仍按原片段审核。</p>
      </div>

      {!editable && (
        <div className="dra-readonly" role="status">
          {!sealed
            ? '当前 Episode 仍在录制，结束并封存后可打标。'
            : '此片段没有可标注帧，请选择非空片段。'}
          已有标注可查看。
        </div>
      )}
      <form onSubmit={save} noValidate>
        <fieldset className="dra-fields" disabled={!editable}>
          <legend className="dra-sr-only">{editingId ? '编辑现有标注' : '新建片段标注'}</legend>
          <div className="dra-form-top">
            <strong>{editingId ? '编辑标注' : '新标注'}</strong>
            <span>
              <Crosshair size={11} />
              当前帧 <b>{Number.isFinite(cursor) ? cursor : '—'}</b>
            </span>
          </div>
          <div className="dra-range-fields">
            <label htmlFor={`${id}-start`}>
              起始帧
              <input
                id={`${id}-start`}
                aria-label="标注起始帧"
                type="number"
                inputMode="numeric"
                step="1"
                min={segment.startFrame}
                max={segment.endFrame - 1}
                value={startFrame}
                aria-invalid={boundaryError && dirty}
                aria-describedby={`${id}-validation`}
                onChange={(event) => {
                  setStartFrame(event.target.value);
                  changed();
                }}
              />
            </label>
            <label htmlFor={`${id}-end`}>
              终止帧 <small>不含</small>
              <input
                id={`${id}-end`}
                aria-label="标注终止帧（不含）"
                type="number"
                inputMode="numeric"
                step="1"
                min={segment.startFrame + 1}
                max={segment.endFrame}
                value={endFrame}
                aria-invalid={boundaryError && dirty}
                aria-describedby={`${id}-validation`}
                onChange={(event) => {
                  setEndFrame(event.target.value);
                  changed();
                }}
              />
            </label>
          </div>
          <div className="dra-range-actions">
            <button
              type="button"
              disabled={!cursorWithinSegment}
              onClick={() => {
                if (!cursorWithinSegment) return;
                setStartFrame(String(cursor));
                changed();
              }}
            >
              当前帧设为起点
            </button>
            <button
              type="button"
              disabled={!cursorWithinSegment}
              title="将终止帧设为当前帧 + 1，以包含当前画面。"
              onClick={() => {
                if (!cursorWithinSegment) return;
                setEndFrame(String(cursor + 1));
                changed();
              }}
            >
              当前帧设为终点
            </button>
          </div>
          <p className="dra-range-help">
            {cursorWithinSegment
              ? '范围 [起点, 终点)，终点按钮取游标 + 1，包含当前帧。'
              : '游标不在当前片段内，请先在画面时间线上定位。'}
          </p>

          <div className="dra-field-group">
            <span id={`${id}-outcome`} className="dra-field-label">
              任务结果
            </span>
            <div className="dra-outcomes" role="group" aria-labelledby={`${id}-outcome`}>
              {(Object.keys(outcomes) as Outcome[]).map((value) => (
                <button
                  type="button"
                  key={value}
                  aria-pressed={outcome === value}
                  className={`${value} ${outcome === value ? 'selected' : ''}`}
                  onClick={() => {
                    setOutcome(value);
                    changed();
                  }}
                >
                  {outcome === value && <Check size={11} />}
                  {outcomes[value]}
                </button>
              ))}
            </div>
          </div>
          <div className="dra-field-group">
            <span id={`${id}-issues`} className="dra-field-label">
              问题标签 <small>可多选</small>
            </span>
            <div className="dra-issue-options" role="group" aria-labelledby={`${id}-issues`}>
              {(Object.keys(issues) as Issue[]).map((value) => (
                <button
                  type="button"
                  key={value}
                  aria-pressed={selectedIssues.includes(value)}
                  className={selectedIssues.includes(value) ? 'selected' : ''}
                  onClick={() => {
                    setSelectedIssues((current) =>
                      current.includes(value)
                        ? current.filter((issue) => issue !== value)
                        : [...current, value],
                    );
                    changed();
                  }}
                >
                  {selectedIssues.includes(value) && <Check size={10} />}
                  {issues[value]}
                </button>
              ))}
            </div>
          </div>
          <label className="dra-field-group" htmlFor={`${id}-intent`}>
            <span className="dra-field-label">纠正意图</span>
            <select
              id={`${id}-intent`}
              aria-label="纠正意图"
              value={intent}
              onChange={(event) => {
                setIntent(event.target.value as Intent);
                changed();
              }}
            >
              {(Object.keys(intents) as Intent[]).map((value) => (
                <option key={value} value={value}>
                  {intents[value]}
                </option>
              ))}
            </select>
          </label>
          <label className="dra-field-group" htmlFor={`${id}-note`}>
            <span className="dra-field-label">
              备注 <small>{note.length} / 240</small>
            </span>
            <textarea
              id={`${id}-note`}
              aria-label="标注备注"
              maxLength={240}
              rows={3}
              value={note}
              placeholder="描述观察、失败原因或纠正目标"
              onChange={(event) => {
                setNote(event.target.value);
                changed();
              }}
            />
          </label>
          <p
            id={`${id}-validation`}
            className={`dra-validation ${validationError && dirty ? 'error' : ''}`}
            aria-live="polite"
          >
            {validationError ||
              `选中 ${input.endFrame - input.startFrame} 帧，可以保存此条语义标注。`}
          </p>
          <div className="dra-save-row">
            <button
              className="dra-save-button"
              type="submit"
              disabled={!editable || !!validationError}
            >
              <Save size={12} />
              {editingId ? '保存修改' : '保存标注'}
            </button>
            <button
              className="dra-new-button"
              type="button"
              onClick={() => {
                resetDraft();
                setFeedback('已切换为新标注，范围恢复为整个片段。');
              }}
            >
              <Plus size={12} />
              新标注
            </button>
          </div>
        </fieldset>
      </form>
      {feedback && (
        <div className="dra-feedback" role="status">
          <CheckCircle2 size={12} />
          {feedback}
        </div>
      )}

      <div className="dra-existing">
        <div className="dra-existing-heading">
          <strong>
            已有标注 <span>{segment.annotations.length}</span>
          </strong>
          <small>当前片段</small>
        </div>
        {segment.annotations.length ? (
          <ol className="dra-annotation-list">
            {segment.annotations.map((annotation, index) => (
              <li key={annotation.id} className={editingId === annotation.id ? 'editing' : ''}>
                <div className="dra-annotation-top">
                  <button
                    className="dra-range-link"
                    type="button"
                    aria-label={`回放标注 ${index + 1}，从第 ${annotation.startFrame} 帧开始`}
                    onClick={() => onSeek(annotation.startFrame)}
                  >
                    <PlayRange />
                    <span>{rangeLabel(annotation)}</span>
                  </button>
                  <div>
                    <button
                      className="dra-icon-button"
                      type="button"
                      disabled={!editable}
                      aria-label={`编辑标注 ${index + 1}`}
                      onClick={() => edit(annotation)}
                    >
                      <Pencil size={12} />
                    </button>
                    <button
                      className="dra-icon-button delete"
                      type="button"
                      disabled={!editable}
                      aria-label={`删除标注 ${index + 1}`}
                      onClick={() => {
                        onRemove(annotation.id);
                        if (editingId === annotation.id) resetDraft();
                        setFeedback('已删除此条本地标注，片段和专家标签有效性不变。');
                      }}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
                <AnnotationLabels annotation={annotation} />
                {annotation.note && <p>{annotation.note}</p>}
              </li>
            ))}
          </ol>
        ) : (
          <div className="dra-no-annotations">
            <Tag size={16} />
            <span>先为这个片段留下一条观察</span>
            <small>可分别标注不同帧范围，支持多条记录。</small>
          </div>
        )}
      </div>
      <footer className="dra-footer">标签不改变专家有效性；是否回流由审核结果决定。</footer>
    </section>
  );
}

function PlayRange() {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
      <path d="M3 2.2 7 5 3 7.8V2.2Z" fill="currentColor" />
    </svg>
  );
}
