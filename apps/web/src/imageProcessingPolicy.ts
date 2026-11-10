import { inferImageLayout, type ImageKind, type ImageLayoutFeature } from './imageLayout';

export type ImageAugmentation = 'default' | 'none' | 'color' | 'crop_rotate_color';

export interface ImageProcessingPolicy {
  resize_mode: 'pad' | 'center_crop' | 'stretch';
  augmentation?: ImageAugmentation;
}

export const imageResizeLabels: Record<ImageProcessingPolicy['resize_mode'], string> = {
  pad: '等比缩放并补黑边',
  center_crop: '中心裁剪后缩放',
  stretch: '直接拉伸',
};

export const imageAugmentationLabels: Record<ImageAugmentation, string> = {
  default: 'OpenPI 默认增强',
  none: '关闭训练增强',
  color: '仅颜色扰动',
  crop_rotate_color: '95% 随机裁剪、±5° 旋转和颜色扰动',
};

export function isImageAugmentation(value: unknown): value is ImageAugmentation {
  return typeof value === 'string' && Object.hasOwn(imageAugmentationLabels, value);
}

export function imageProcessingErrors(value: unknown): string[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ['图像处理配置无效。'];
  const row = value as Record<string, unknown>;
  if (
    !Object.keys(row).every((key) => ['resize_mode', 'augmentation'].includes(key)) ||
    typeof row.resize_mode !== 'string' ||
    !Object.hasOwn(imageResizeLabels, row.resize_mode) ||
    (Object.hasOwn(row, 'augmentation') && !isImageAugmentation(row.augmentation))
  )
    return ['图像缩放方式或历史增强配置无效。'];
  return [];
}

export function imageSourceResolution(
  features: readonly ImageLayoutFeature[] | undefined,
  key: string,
  kind: ImageKind,
): [number, number] | undefined {
  const inferred = inferImageLayout(features, key, kind);
  const shape = inferred.frameShape;
  if (!shape) return undefined;
  if (inferred.depthWithoutChannel) return [shape[0], shape[1]];
  if (!inferred.layout) return undefined;
  return inferred.layout === 'chw' ? [shape[1], shape[2]] : [shape[0], shape[1]];
}

interface CameraPolicy {
  key: string;
  kind: ImageKind;
  image_processing?: ImageProcessingPolicy;
}

/** Registration must use real source metadata; unknown layouts never imply 224×224. */
export function imageRegistrationErrors(
  cameras: readonly CameraPolicy[] | undefined,
  features: readonly ImageLayoutFeature[] | undefined,
): string[] {
  if (!features) return ['请先读取数据集元信息，确认各路图像尺寸后再登记。'];
  const errors: string[] = [];
  const keys = new Set<string>();
  for (const feature of features.filter((row) => ['image', 'video'].includes(row.dtype))) {
    const camera = cameras?.find((row) => row.key.trim() === feature.key);
    const kind =
      camera?.kind || (/(?:^|[._/])depth(?:$|[._/])/i.test(feature.key) ? 'depth' : 'rgb');
    const size = imageSourceResolution(features, feature.key, kind);
    keys.add(feature.key);
    if (!size && !camera?.image_processing)
      errors.push(
        `图像字段 ${feature.key} 的尺寸或布局无法确认，请明确选择图像处理方式并在训练配置中核对布局。`,
      );
    else if (size && (size[0] !== 224 || size[1] !== 224) && !camera?.image_processing)
      errors.push(
        `图像字段 ${feature.key} 为 ${size[1]}×${size[0]}，请登记该视觉输入并明确选择处理到 224×224 的方式。`,
      );
  }
  for (const camera of cameras || []) {
    if (camera.image_processing) errors.push(...imageProcessingErrors(camera.image_processing));
    if (
      camera.kind === 'depth' &&
      camera.image_processing?.augmentation !== undefined &&
      !['default', 'none'].includes(camera.image_processing.augmentation)
    )
      errors.push(`深度字段 ${camera.key} 不支持 RGB 颜色增强，请选择默认增强或关闭增强。`);
    if (keys.has(camera.key.trim())) continue;
    const size = imageSourceResolution(features, camera.key, camera.kind);
    if (!features.some((feature) => feature.key === camera.key.trim()))
      errors.push(`视觉字段 ${camera.key} 缺少可确认的图像尺寸，请重新读取元信息。`);
    else if (!size && !camera.image_processing)
      errors.push(`视觉字段 ${camera.key} 的尺寸或布局无法确认，请明确选择图像处理方式。`);
    else if (size && (size[0] !== 224 || size[1] !== 224) && !camera.image_processing)
      errors.push(`视觉字段 ${camera.key} 不是 224×224，请明确选择图像处理方式。`);
  }
  return errors;
}

export function imageProcessingSummary(value: ImageProcessingPolicy): string {
  return `${imageResizeLabels[value.resize_mode]} → 224×224`;
}

/** Use template geometry and preserve legacy augmentation only on an existing matching source. */
export function retainMatchingImageProcessing<T extends { cameras: readonly CameraPolicy[] }>(
  template: T,
  current: T,
): T {
  return {
    ...structuredClone(template),
    cameras: template.cameras.map(({ image_processing, ...camera }) => {
      const previous = current.cameras.find(
        (row) => row.key.trim() === camera.key.trim() && row.kind === camera.kind,
      )?.image_processing;
      const resize_mode = image_processing?.resize_mode ?? previous?.resize_mode;
      const augmentation = previous?.augmentation;
      return {
        ...structuredClone(camera),
        ...(resize_mode
          ? {
              image_processing: {
                resize_mode,
                ...(augmentation !== undefined ? { augmentation } : {}),
              },
            }
          : {}),
      };
    }),
  };
}
