import { FolderPlus } from 'lucide-react';
import { assetTargetNameError } from './assetTransfer';
import type { AssetSyncDirectoryDraft } from './assetTransfer';
import { Field } from './components';
import { PathAutocomplete } from './PathAutocomplete';
import type { Node } from './types';

export function AssetSyncDirectory({
  draft,
  onChange,
  node,
  busy,
}: {
  draft: AssetSyncDirectoryDraft;
  onChange: (draft: AssetSyncDirectoryDraft) => void;
  node?: Node;
  busy: boolean;
}) {
  const creating = draft.newFolder !== undefined;
  const disabled = busy || node?.status !== 'ONLINE';
  return (
    <div className="asset-sync-directory">
      <PathAutocomplete
        node={node}
        label={creating ? '上一级目录' : '目标父目录'}
        value={draft.parent}
        onChange={(parent) => onChange({ ...draft, parent })}
        disabled={disabled}
        placeholder="例如 /data/shared"
        hint={
          creating
            ? '选择或输入新文件夹所在的上一级目录。'
            : '可选择已有目录，也可直接填写完整路径。托管目录下缺失的文件夹会在同步时创建。'
        }
      />
      <button
        type="button"
        className="button outline small"
        aria-expanded={creating}
        disabled={disabled}
        onClick={() => onChange({ parent: draft.parent, newFolder: creating ? undefined : '' })}
      >
        {!creating && <FolderPlus size={15} />}
        {creating ? '取消新建文件夹' : '新建文件夹'}
      </button>
      {creating && (
        <Field label="新文件夹名称" required hint="同步时创建，资产将复制到此文件夹下。">
          <input
            aria-label="新文件夹名称"
            className="mono"
            placeholder="例如 checkpoints"
            value={draft.newFolder}
            maxLength={160}
            disabled={disabled}
            aria-invalid={Boolean(draft.newFolder && assetTargetNameError(draft.newFolder))}
            onChange={(event) => onChange({ ...draft, newFolder: event.target.value })}
          />
        </Field>
      )}
    </div>
  );
}
