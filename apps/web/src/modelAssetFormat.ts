import type { Asset } from './types';

export const modelFormatLabels = {
  'openpi-orbax': 'JAX / Orbax',
  'openpi-pytorch': 'PyTorch / Safetensors',
} as const;

export function modelAssetFormat(asset: Asset): keyof typeof modelFormatLabels | undefined {
  const format = asset.metadata?.format;
  if (typeof format !== 'string') return undefined;
  const normalized = format.trim().toLowerCase();
  if (['openpi-orbax', 'orbax', 'jax', 'openpi-jax'].includes(normalized)) return 'openpi-orbax';
  if (['openpi-pytorch', 'openpi-torch', 'pytorch', 'torch'].includes(normalized))
    return 'openpi-pytorch';
  return undefined;
}
