import { useState } from 'react';
import { useAuth } from './auth';
import { Modal } from './components';
import { DatasetFormatTemplates } from './DatasetFormatTemplates';
import {
  DatasetProfileForm,
  datasetProfileErrors,
  emptyDatasetProfile,
} from './DatasetProfileForm';

export function PersonalDatasetFormats({ close }: { close: () => void }) {
  const { user, can } = useAuth();
  if (!user?.id || !can('dataset-templates.write'))
    return (
      <Modal title="权限不足" close={close}>
        管理个人格式模板需要操作员或管理员账号。
      </Modal>
    );
  return <PersonalDatasetFormatsContent key={user.id} close={close} />;
}

function PersonalDatasetFormatsContent({ close }: { close: () => void }) {
  const [profile, setProfile] = useState(emptyDatasetProfile);
  const errors = datasetProfileErrors(profile);
  return (
    <Modal
      title="我的格式模板"
      wide
      className="personal-dataset-formats"
      close={close}
      footer={
        <button className="button outline" type="button" onClick={close}>
          关闭
        </button>
      }
    >
      <p className="field-hint">
        模板仅当前账号可见。载入模板只填写下方个人草稿；修改后可覆盖同名模板，或换一个名称保存副本。
        此窗口不会登记或修改公共数据集，公共数据集的格式维护由管理员完成。
      </p>
      <DatasetFormatTemplates value={profile} onApply={setProfile} personalDraft />
      <section aria-label="个人格式草稿">
        <h3>个人格式草稿</h3>
        <p className="field-hint">未保存的草稿仅保留在当前窗口；保存为个人模板后可跨设备使用。</p>
        <DatasetProfileForm value={profile} onChange={setProfile} compact />
        {errors.length > 0 && <p className="field-hint">保存前请完善：{errors.join('；')}</p>}
      </section>
    </Modal>
  );
}
