export type ImageLayout = 'chw' | 'hwc';
export type ImageKind = 'rgb' | 'depth';

export interface ImageLayoutEdits {
  has: (marker: string) => boolean;
  set: (marker: string, manual: boolean) => void;
}

/** Bounded draft key for one binding/camera, independent of component mounting. */
export function imageLayoutManualMarker(scope: readonly unknown[]): string {
  let first = 0x811c9dc5, second = 0x9e3779b9;
  const text = JSON.stringify(scope);
  for (let index = 0; index < text.length; index++) {
    first = Math.imul(first ^ text.charCodeAt(index), 0x01000193);
    second = Math.imul(second ^ text.charCodeAt(index), 0x85ebca6b);
  }
  return `image_layout.manual:${(first >>> 0).toString(16).padStart(8, '0')}${(second >>> 0).toString(16).padStart(8, '0')}`;
}

export interface ImageLayoutFeature {
  key: string;
  dtype: string;
  shape: readonly number[];
  names?: readonly string[];
}

export type ImageFeature = ImageLayoutFeature;

export interface ImageLayoutCamera {
  key: string;
  kind: ImageKind;
  layout: ImageLayout | '';
  time_index?: number | null;
}

export interface ImageLayoutInference {
  layout?: ImageLayout;
  shape?: number[];
  frameShape?: number[];
  requiresTimeIndex: boolean;
  timeLength?: number;
  depthWithoutChannel?: boolean;
  reason?: string;
}

function sourceImage(features: readonly ImageLayoutFeature[] | undefined, key: string) {
  if (!Array.isArray(features) || !features.length)
    return {
      reason: '尚未读取数据集图像字段形状，请读取元信息或手动确认布局。',
      unavailable: true,
    };
  const matches = (features as readonly ImageLayoutFeature[]).filter(
    (feature) => typeof feature?.key === 'string' && feature.key.trim() === key.trim(),
  );
  if (matches.length !== 1)
    return { reason: `视觉字段 ${key || '（未选择）'} 没有唯一的元信息，请核对字段或重新读取。` };
  const feature = matches[0];
  // This function is called for a declared camera input. LeRobot depth and
  // explicitly registered RGB arrays may use numeric features rather than a codec.
  if (
    !['image', 'video'].includes(feature.dtype) &&
    !/^(?:u?int(?:8|16|32|64)?|float(?:16|32|64)?|double)$/.test(feature.dtype)
  )
    return { reason: `视觉字段 ${key} 的真实类型不是图像或数值张量。` };
  if (
    !Array.isArray(feature.shape) ||
    !feature.shape.length ||
    feature.shape.some((size) => !Number.isSafeInteger(size) || size < 1 || size > 2 ** 31 - 1)
  )
    return { reason: `视觉字段 ${key} 的原始形状无效，无法判断布局。` };
  return { shape: [...feature.shape] };
}

/** Infer only a unique channel axis; never pick an image frame or alter a camera. */
export function inferImageLayout(
  features: readonly ImageLayoutFeature[] | undefined,
  key: string,
  kind: ImageKind,
): ImageLayoutInference {
  const source = sourceImage(features, key);
  if (!source.shape) return { requiresTimeIndex: false, reason: source.reason };
  const shape = source.shape;
  const requiresTimeIndex = shape.length === 4;
  const frameShape = requiresTimeIndex ? shape.slice(1) : [...shape];
  const result: ImageLayoutInference = {
    shape,
    frameShape,
    requiresTimeIndex,
    ...(requiresTimeIndex ? { timeLength: shape[0] } : {}),
  };
  if (kind === 'depth' && frameShape.length === 2)
    return {
      ...result,
      depthWithoutChannel: true,
      reason: '二维深度图没有通道轴，CHW/HWC 不影响该字段；不自动推断布局。',
    };
  if (frameShape.length !== 3)
    return { ...result, reason: '图像形状不属于受支持的单帧或单个额外首轴形式，当前不推断布局。' };
  const channels = kind === 'depth' ? 1 : 3;
  const first = frameShape[0] === channels,
    last = frameShape[2] === channels;
  if (first === last)
    return {
      ...result,
      reason: first
        ? '首尾维度均可能是通道轴，无法唯一判断 CHW/HWC，请手动确认。'
        : `没有唯一的 ${channels} 通道轴；请核对图像类型、布局，或在首轴确为历史时显式选择图像时间索引。`,
    };
  return {
    ...result,
    layout: first ? 'chw' : 'hwc',
    ...(requiresTimeIndex
      ? { reason: '图像包含额外首轴；仅推断单帧布局，仍需显式选择图像时间索引，不会自动取帧。' }
      : {}),
  };
}

/** Validate known source facts early; unavailable inspection does not invent a blocker. */
export function imageLayoutErrors(
  camera: ImageLayoutCamera,
  features: readonly ImageLayoutFeature[] | undefined,
): string[] {
  const source = sourceImage(features, camera.key);
  if (!source.shape) return source.unavailable ? [] : [source.reason!];
  const shape = source.shape;
  const prefix = `视觉字段 ${camera.key}：`;
  const selected = camera.time_index !== undefined && camera.time_index !== null;
  if (
    selected &&
    (!Number.isSafeInteger(camera.time_index) ||
      camera.time_index! < -64 ||
      camera.time_index! > 63)
  )
    return [prefix + '图像时间索引必须是 -64 至 63 的整数。'];
  if (selected && (camera.time_index! < -shape[0] || camera.time_index! >= shape[0]))
    return [prefix + `图像时间索引越界，首轴长度为 ${shape[0]}。`];
  if (selected && (camera.kind === 'rgb' ? shape.length !== 4 : ![3, 4].includes(shape.length)))
    return [prefix + '当前形状没有可用的额外首轴，请清空图像时间索引并核对原始形状。'];
  if (!selected && shape.length === 4)
    return [
      prefix + '原始图像包含额外首轴，请显式填写图像时间索引；不会自动选择第一帧或最后一帧。',
    ];
  if (!['chw', 'hwc'].includes(camera.layout)) return [prefix + '请明确选择 CHW 或 HWC 布局。'];
  const frame = selected ? shape.slice(1) : shape;
  const channels = frame[camera.layout === 'chw' ? 0 : frame.length - 1];
  if (camera.kind === 'rgb' && (frame.length !== 3 || channels !== 3))
    return [
      prefix +
        `形状 [${frame.join(', ')}] 与 ${camera.layout.toUpperCase()} 的 RGB 三通道配置不一致。`,
    ];
  if (camera.kind === 'depth' && !(frame.length === 2 || (frame.length === 3 && channels === 1)))
    return [
      prefix + '深度图必须是二维或具有明确的单通道轴；如首轴为历史，请显式选择图像时间索引。',
    ];
  return [];
}
