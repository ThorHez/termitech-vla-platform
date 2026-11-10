import type { DatasetProfile } from './DatasetProfileForm';
import { imageProcessingSummary } from './imageProcessingPolicy';

export function DatasetImageProcessingSummary({ profile }: { profile?: DatasetProfile }) {
  if (!profile) return null;
  return (
    <div className="dataset-image-processing-summary">
      <strong>图像处理 · 目标 224×224</strong>
      {profile.cameras.map((camera) => (
        <p className="field-hint" key={camera.key}>
          {camera.name || camera.key}：
          {camera.image_processing
            ? imageProcessingSummary(camera.image_processing)
            : '未单独登记处理策略'}
        </p>
      ))}
      <p className="field-hint">
        缩放方式来自数据集资产。逐路训练增强在本次训练的组合视觉输入中设置。
      </p>
    </div>
  );
}
