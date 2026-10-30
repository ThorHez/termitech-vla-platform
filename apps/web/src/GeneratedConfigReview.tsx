import { useState } from 'react';
import { Copy, Download, FileCode2 } from 'lucide-react';
import { configurationConfirmed, generatedConfigAvailable } from './trainingConfiguration';
import type { GeneratedTrainingConfig } from './types';
import './generated-config-review.css';

export function GeneratedConfigReview({
  config,
  confirmedSha256,
  onConfirm,
  readOnly = false,
}: {
  config?: GeneratedTrainingConfig;
  confirmedSha256?: string;
  onConfirm?: (sha256: string | undefined) => void;
  readOnly?: boolean;
}) {
  const [copyState, setCopyState] = useState('复制代码');
  if (!generatedConfigAvailable(config))
    return (
      <div className="inline-note amber" role="alert">
        预检尚未提供本次训练的 Python config，暂时无法确认启动。请更新中台、Agent
        和训练适配器后重新预检。
      </div>
    );
  const download = () => {
    const url = URL.createObjectURL(
      new Blob([config.source], { type: 'text/x-python;charset=utf-8' }),
    );
    const link = document.createElement('a');
    link.href = url;
    link.download = 'training_config.py';
    link.click();
    URL.revokeObjectURL(url);
  };
  return (
    <section
      className={`generated-config-review${readOnly ? ' generated-config-readonly' : ''}`}
      aria-label={readOnly ? '本次训练 config' : '检查本次训练 config'}
    >
      <div className="generated-config-heading">
        <div>
          <h3>{readOnly ? '本次训练 config' : '检查本次训练 config'}</h3>
          <p>
            {readOnly
              ? '由 Agent 根据本次实际解析结果生成，保留模型、数据处理、资产路径和训练参数。'
              : '请核对 Agent 生成的模型、数据处理、资产路径和训练参数。'}
          </p>
        </div>
        <div className="generated-config-actions">
          <button
            type="button"
            className="button outline small"
            onClick={() =>
              void navigator.clipboard
                .writeText(config.source)
                .then(() => setCopyState('已复制'))
                .catch(() => setCopyState('复制失败，请下载'))
            }
          >
            <Copy size={14} />
            {copyState}
          </button>
          <button type="button" className="button outline small" onClick={download}>
            <Download size={14} />
            下载 config
          </button>
        </div>
      </div>
      <div className="generated-config-body">
        <div className="generated-config-code">
          <div className="generated-config-file">
            <span>
              <FileCode2 size={14} aria-hidden="true" />
              training_config.py
            </span>
            <span>Python</span>
          </div>
          <pre className="generated-config-source" tabIndex={0} aria-label="Python 配置代码">
            <code>{config.source}</code>
          </pre>
        </div>
        <p className="generated-config-digest">
          <span>SHA-256</span>
          <code>{config.sha256}</code>
        </p>
        {!readOnly && (
          <label className="generated-config-confirmation">
            <input
              type="checkbox"
              checked={configurationConfirmed(config, confirmedSha256)}
              onChange={(event) => onConfirm?.(event.target.checked ? config.sha256 : undefined)}
            />
            <span>已核对本次 config，确认启动训练</span>
          </label>
        )}
        {!readOnly && (
          <p className="field-hint">
            修改配置或重新预检后需要重新确认；启动前会再次核对代码与资产。
          </p>
        )}
      </div>
    </section>
  );
}
