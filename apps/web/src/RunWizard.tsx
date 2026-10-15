import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  Circle,
  Cpu,
  Database,
  GitCommitHorizontal,
  LoaderCircle,
  Play,
  RotateCcw,
  Server,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import { navigate, post, useApi } from './api';
import { CodeVersion, ErrorNotice, Field, Modal } from './components';
import { inputRoles, requiredInputs, useData } from './data';
import type { ParameterSpec, Preview, Run, RunRequest, Template } from './types';

function specs(template?: Template): Record<string, ParameterSpec> {
  if (Array.isArray(template?.parameters))
    return Object.fromEntries(
      template.parameters.filter((value) => value.name).map((value) => [value.name!, value]),
    );
  return Object.fromEntries(
    Object.entries(template?.parameters || {}).map(([key, value]) => [
      key,
      typeof value === 'number' ? { default: value, type: 'number' } : value,
    ]),
  );
}
export default function RunWizard({
  close,
  initialNode,
  initialTemplate,
}: {
  close: () => void;
  initialNode?: string;
  initialTemplate?: string;
}) {
  const { nodes, assets, profiles, templates, refresh } = useData();
  const [step, setStep] = useState(0);
  const [request, setRequest] = useState<RunRequest>({
    name: '',
    node_id: initialNode || '',
    profile_id: '',
    template_id:
      initialTemplate ||
      templates.data?.find((item) => item.id === 'openpi-sft')?.id ||
      templates.data?.find((item) => item.status === 'AVAILABLE')?.id ||
      '',
    inputs: {},
    parameters: {},
    gpu_count: 1,
  });
  const [previewId, setPreviewId] = useState('');
  const [createdPreview, setCreatedPreview] = useState<Preview>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error>();
  const idempotencyKey = useRef(crypto.randomUUID());
  const previewResource = useApi<Preview>(previewId ? `/run-previews/${previewId}` : null, 1800);
  const preview = previewResource.data?.id === previewId ? previewResource.data : createdPreview;
  const template = templates.data?.find((item) => item.id === request.template_id);
  const node = nodes.data?.find((item) => item.id === request.node_id);
  const compatibleProfiles = useMemo(
    () =>
      profiles.data?.filter(
        (profile) =>
          profile.node_id === request.node_id &&
          profile.adapter === request.template_id &&
          (!profile.execution_mode || profile.execution_mode === 'native'),
      ) || [],
    [profiles.data, request.node_id, request.template_id],
  );
  const parameterSpecs = specs(template);
  const selectedProfile = compatibleProfiles.find((profile) => profile.id === request.profile_id);
  const minGpus = template?.min_gpus ?? (template?.id === 'smoke' ? 0 : 1);
  useEffect(() => {
    const rules = specs(template);
    setRequest((value) => ({
      ...value,
      parameters: Object.fromEntries(
        Object.entries(rules).map(([key, rule]) => [key, rule.default ?? 1]),
      ),
      gpu_count: template?.min_gpus ?? (template?.id === 'smoke' ? 0 : 1),
      inputs: {},
    }));
    // A change to template metadata does not reset an in-progress form.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.template_id]);
  useEffect(() => {
    setRequest((value) =>
      compatibleProfiles.some((profile) => profile.id === value.profile_id)
        ? value
        : { ...value, profile_id: compatibleProfiles[0]?.id || '' },
    );
  }, [compatibleProfiles]);
  const update = <K extends keyof RunRequest>(key: K, value: RunRequest[K]) =>
    setRequest((previous) => ({ ...previous, [key]: value }));
  const beginPreview = async () => {
    setBusy(true);
    setError(undefined);
    setPreviewId('');
    setCreatedPreview(undefined);
    idempotencyKey.current = crypto.randomUUID();
    try {
      const result = await post<Preview>('/run-previews', request);
      setCreatedPreview(result);
      setPreviewId(result.id);
    } catch (err) {
      setError(err as Error);
    } finally {
      setBusy(false);
    }
  };
  const start = async () => {
    if (preview?.status !== 'READY') return;
    setBusy(true);
    setError(undefined);
    try {
      const run = await post<Run>(
        '/runs',
        { preflight_id: preview.id },
        { 'Idempotency-Key': idempotencyKey.current },
      );
      refresh();
      close();
      navigate(`/runs/${run.id}`);
    } catch (err) {
      setError(err as Error);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal
      title="新建训练"
      subtitle="选择本机资源，预检通过后启动。"
      close={() => {
        if (!busy) close();
      }}
      wide
      footer={
        <>
          <div className="wizard-footer-note">
            <ShieldCheck size={14} />
            <span>
              {step === 2 ? '代码版本由 Agent 读取并绑定本次训练' : '数据与模型保持在所选机器本地'}
            </span>
          </div>
          {step > 0 && (
            <button
              className="button outline"
              disabled={busy}
              onClick={() => {
                setStep(step - 1);
                setPreviewId('');
                setCreatedPreview(undefined);
                setError(undefined);
              }}
            >
              <ArrowLeft size={15} />
              上一步
            </button>
          )}
          {step < 2 ? (
            <button className="button primary" type="submit" form="run-wizard-form">
              {step === 1 ? '开始预检' : '下一步'}
              <ArrowRight size={15} />
            </button>
          ) : (
            <button
              className="button primary"
              disabled={busy || preview?.status !== 'READY'}
              onClick={() => void start()}
            >
              {busy ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}确认启动训练
            </button>
          )}
        </>
      }
    >
      <div className="wizard-steps">
        {[
          { title: '机器与模板', icon: Server },
          { title: '资产与参数', icon: Database },
          { title: '预检与确认', icon: GitCommitHorizontal },
        ].map((item, index) => (
          <div
            key={item.title}
            className={`wizard-step ${index === step ? 'current' : index < step ? 'done' : ''}`}
          >
            <span>{index < step ? <Check size={15} /> : index + 1}</span>
            <strong>{item.title}</strong>
          </div>
        ))}
      </div>
      <ErrorNotice error={error || (step === 2 ? previewResource.error : undefined)} />
      {step < 2 && (
        <form
          id="run-wizard-form"
          onSubmit={(event) => {
            event.preventDefault();
            if (
              step === 0 &&
              (!selectedProfile || node?.status !== 'ONLINE' || template?.status !== 'AVAILABLE')
            ) {
              setError(new Error('请选择在线机器、可用模板和匹配的本机运行环境。'));
              return;
            }
            setError(undefined);
            setStep(step + 1);
            if (step === 1) void beginPreview();
          }}
        >
          {step === 0 ? (
            <>
              <Field label="训练名称" required hint="使用清晰的名称，便于查找训练结果。">
                <input
                  required
                  autoFocus
                  maxLength={160}
                  placeholder="例如 shellgame-sft-1015"
                  value={request.name}
                  onChange={(event) => update('name', event.target.value)}
                />
              </Field>
              <div className="form-grid">
                <Field label="计算机器" required>
                  <select
                    required
                    value={request.node_id}
                    onChange={(event) => {
                      update('node_id', event.target.value);
                      update('inputs', {});
                    }}
                  >
                    <option value="">选择在线机器</option>
                    {nodes.data?.map((item) => (
                      <option key={item.id} value={item.id} disabled={item.status !== 'ONLINE'}>
                        {item.name} · {item.gpus?.length || 0} GPU
                        {item.status !== 'ONLINE' ? ' · 离线' : ''}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="训练模板" required>
                  <select
                    required
                    value={request.template_id}
                    onChange={(event) => update('template_id', event.target.value)}
                  >
                    <option value="">选择模板</option>
                    {templates.data?.map((item) => (
                      <option key={item.id} value={item.id} disabled={item.status !== 'AVAILABLE'}>
                        {item.name}
                        {item.status !== 'AVAILABLE' ? ' · 待验收' : ''}
                      </option>
                    ))}
                  </select>
                </Field>
              </div>
              {template && (
                <div className={`template-note ${template.id === 'smoke' ? 'amber' : ''}`}>
                  <ShieldCheck size={17} />
                  <p>{template.description}</p>
                </div>
              )}
              <Field
                label="本机运行环境"
                required
                hint="环境由管理员配置，平台根据模板过滤兼容的环境。"
              >
                <select
                  required
                  value={request.profile_id}
                  onChange={(event) => update('profile_id', event.target.value)}
                >
                  <option value="">
                    {request.node_id
                      ? compatibleProfiles.length
                        ? '选择环境'
                        : '此机器没有匹配的运行环境'
                      : '请先选择机器'}
                  </option>
                  {compatibleProfiles.map((profile) => (
                    <option value={profile.id} key={profile.id}>
                      {profile.name}
                    </option>
                  ))}
                </select>
              </Field>
              {selectedProfile && (
                <dl className="detail-list inset">
                  <div>
                    <dt>代码目录</dt>
                    <dd>
                      <code>{selectedProfile.code_dir}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Python</dt>
                    <dd>
                      <code>{selectedProfile.python}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>执行方式</dt>
                    <dd>{selectedProfile.execution_mode || 'native'}</dd>
                  </div>
                  {selectedProfile.recipe && (
                    <div>
                      <dt>训练配方</dt>
                      <dd>
                        <code>{selectedProfile.recipe}</code>
                      </dd>
                    </div>
                  )}
                </dl>
              )}
            </>
          ) : (
            <>
              <div className="selection-summary">
                <Server size={16} />
                <strong>{node?.name}</strong>
                <span>→</span>
                <span>{template?.name}</span>
              </div>
              {requiredInputs(template).length ? (
                requiredInputs(template).map((role) => {
                  const kind = role === 'dataset' ? 'DATASET' : role === 'model' ? 'MODEL' : 'NORM';
                  const choices =
                    assets.data?.filter(
                      (asset) =>
                        asset.node_id === request.node_id &&
                        asset.kind === kind &&
                        !['ARCHIVED', 'MISSING', 'INVALID'].includes(
                          asset.status || asset.state || '',
                        ),
                    ) || [];
                  return (
                    <Field
                      key={role}
                      label={inputRoles[role] || role}
                      required
                      hint={
                        choices.length
                          ? '只显示当前机器的资产，实际可用性将在预检时验证。'
                          : '当前机器没有可选资产，请先在「数据与模型」中登记。'
                      }
                    >
                      <select
                        required
                        value={request.inputs[role] || ''}
                        onChange={(event) =>
                          update('inputs', { ...request.inputs, [role]: event.target.value })
                        }
                      >
                        <option value="">选择{inputRoles[role] || role}</option>
                        {choices.map((asset) => (
                          <option key={asset.id} value={asset.id}>
                            {asset.name} · {asset.version}
                          </option>
                        ))}
                      </select>
                    </Field>
                  );
                })
              ) : (
                <div className="inline-note">
                  当前模板使用合成数据进行链路测试，无需选择本地数据资产。
                </div>
              )}
              <div className="form-section-title">
                <Cpu size={16} />
                <h3>训练参数</h3>
                <span>模板已填入推荐默认值</span>
              </div>
              <div className="form-grid parameters">
                {Object.entries(parameterSpecs).map(([key, spec]) => (
                  <Field key={key} label={spec.label || key} required>
                    <input
                      type="number"
                      required
                      min={spec.min ?? spec.minimum}
                      max={spec.max ?? spec.maximum}
                      step={spec.type === 'integer' ? 1 : 'any'}
                      value={
                        Number.isFinite(request.parameters[key]) ? request.parameters[key] : ''
                      }
                      onChange={(event) =>
                        update('parameters', {
                          ...request.parameters,
                          [key]: event.target.value === '' ? NaN : Number(event.target.value),
                        })
                      }
                    />
                  </Field>
                ))}
                <Field
                  label="申请 GPU 数量"
                  hint={minGpus === 0 ? '测试模板支持 0 GPU。' : '资源不足时将排队等待。'}
                >
                  <input
                    type="number"
                    required
                    min={minGpus}
                    max={Math.max(minGpus, node?.gpus?.length || 0)}
                    step={1}
                    value={request.gpu_count}
                    onChange={(event) => update('gpu_count', Number(event.target.value))}
                  />
                </Field>
              </div>
            </>
          )}
        </form>
      )}
      {step === 2 && (
        <div className="preflight-content">
          <div
            className={`preflight-heading ${preview?.status === 'READY' ? 'ready' : preview?.status === 'FAILED' ? 'failed' : ''}`}
          >
            {preview?.status === 'READY' ? (
              <CheckCircle2 size={28} />
            ) : preview?.status === 'FAILED' ? (
              <XCircle size={28} />
            ) : busy || preview?.status === 'PENDING' ? (
              <LoaderCircle size={28} className="spin" />
            ) : (
              <Circle size={28} />
            )}
            <div>
              <h3>
                {preview?.status === 'READY'
                  ? '预检通过，可以开始训练'
                  : preview?.status === 'FAILED'
                    ? '预检未通过'
                    : busy || preview?.status === 'PENDING'
                      ? 'Agent 正在检查本机环境'
                      : '等待完成预检'}
              </h3>
              <p>
                {preview?.status === 'READY'
                  ? '启动前还会核对代码版本与资源占用。'
                  : '检查资产路径、运行环境、GPU 和实际执行代码。'}
              </p>
            </div>
          </div>
          {preview?.checks?.length ? (
            <div className="preflight-checks">
              {preview.checks.map((check, index) => (
                <div className={check.ok ? 'ok' : 'bad'} key={`${check.name}-${index}`}>
                  {check.ok ? <CheckCircle2 size={17} /> : <XCircle size={17} />}
                  <strong>{check.name}</strong>
                  <span>{check.message}</span>
                </div>
              ))}
            </div>
          ) : (
            preview?.status === 'PENDING' && (
              <div className="preflight-wait">
                等待机器回传预检结果。机器在线时会自动继续；可关闭弹窗后重新发起。
              </div>
            )
          )}
          {preview?.error && (
            <div className="inline-note red">
              <strong>{preview.error.code}</strong>
              <span>{preview.error.message}</span>
            </div>
          )}
          {preview?.code_snapshot && (
            <CodeVersion snapshot={preview.code_snapshot} title="本次训练代码基线" />
          )}
          <dl className="detail-list inset">
            <div>
              <dt>训练名称</dt>
              <dd>{request.name}</dd>
            </div>
            <div>
              <dt>机器 / 模板</dt>
              <dd>
                {node?.name} / {template?.name}
              </dd>
            </div>
            <div>
              <dt>申请 GPU</dt>
              <dd>{request.gpu_count} 张</dd>
            </div>
          </dl>
          {(!preview || ['FAILED', 'EXPIRED'].includes(preview.status)) && !busy && (
            <button className="button outline" onClick={() => void beginPreview()}>
              <RotateCcw size={15} />
              重新预检
            </button>
          )}
        </div>
      )}
    </Modal>
  );
}
