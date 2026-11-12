import { navigate } from './api';
import { ArrowRight, ArrowUpRight, Box, Database, Workflow } from 'lucide-react';
import { Badge, Modal, PageHeading } from './components';
import { useData } from './data';
import { sceneBindings } from './scenarioBindings';
import './scenario-showcase.css';

const featuredScene = {
  id: 'components',
  title: '电子元器件分拣',
  description: '抓取电子元器件并按类别归置。',
  task: '抓取电子元器件，并按类别放入对应区域。',
  image: new URL('./scenario-images/components.jpg', import.meta.url).href,
  imageLabel: '机械臂夹起电路板，旁边是用于分拣的料盒',
  data: '登记元器件分拣的示教数据，校验观测和动作输入。',
  training: '运行元器件抓取与分类放置的策略训练。',
  validation: '部署策略，检查元器件的抓取和分类放置结果。',
};

const scenes = [
  {
    id: 'fold',
    title: '双臂叠衣服',
    description: '双臂协同抓取、展平并折叠衣物。',
    task: '使用 ARX5 双臂协同抓取、展平并折叠衣物。',
    image: new URL('./scenario-images/fold.jpg', import.meta.url).href,
    imageLabel: '两台机械臂抓住一件蓝色衣物',
    data: '登记衣物操作的双臂示教数据，校验训练输入。',
    training: '跟踪叠衣策略的训练任务和模型检查点。',
    validation: '部署策略，观察衣物对齐和折叠效果。',
  },
  featuredScene,
  {
    id: 'tray',
    title: '料盘抓放',
    description: '抓取卷带式圆形料盘并放到指定位置。',
    task: '机器人抓取卷带式圆形料盘，将整盘搬运到指定位置并放下。',
    image: new URL('./scenario-images/tray.jpg', import.meta.url).href,
    imageLabel: '机械夹爪抓住带中心孔和辐条的圆形卷带料盘',
    data: '登记卷带式料盘抓取、搬运和放置的示教数据。',
    training: '训练料盘抓放策略，管理模型检查点。',
    validation: '部署策略，检查抓取稳定性和料盘落位。',
  },
  {
    id: 'goods',
    title: '商品分拣',
    description: '将商品分拣到货架上的对应位置。',
    task: '机器人按分拣规则抓取商品，并放到货架上的对应位置。',
    image: new URL('./scenario-images/goods.jpg', import.meta.url).href,
    imageLabel: '机械臂将包装盒放到货架上',
    data: '登记不同商品的抓取与上架示教数据。',
    training: '训练商品分拣策略，跟踪任务和模型检查点。',
    validation: '部署策略，检查商品是否放到对应的货架位置。',
  },
];

type Scene = (typeof scenes)[number];

function sceneDetailHref(sceneId: string) {
  return `#/scenes?scene=${encodeURIComponent(sceneId)}`;
}

function sceneAssetsHref(sceneId: string, assetId?: string) {
  const scene = `#/assets?scene=${encodeURIComponent(sceneId)}`;
  return assetId ? `${scene}&asset=${encodeURIComponent(assetId)}` : scene;
}

export default function ScenarioShowcase({ sceneId }: { sceneId?: string }) {
  const selectedScene = scenes.find((scene) => scene.id === sceneId);

  return (
    <div className="scenario-showcase">
      <PageHeading
        title="场景展示"
        description="查看已接入平台的机器人场景及其关联的训练任务、数据集和模型。"
      />
      <section className="scenario-hero" aria-labelledby="scenario-featured-title">
        <div className="scenario-hero-copy">
          <span className="scenario-feature-label">重点场景</span>
          <h2 id="scenario-featured-title">{featuredScene.title}</h2>
          <p>机械臂抓取电子元器件并按类别归置。平台集中管理分拣示教数据、策略训练和模型检查点。</p>
          <div className="scenario-hero-actions">
            <a
              className="button primary scenario-hero-action"
              href={sceneAssetsHref(featuredScene.id)}
            >
              查看数据与模型 <ArrowRight size={17} aria-hidden="true" />
            </a>
            <a
              className="button scenario-hero-detail"
              href={sceneDetailHref(featuredScene.id)}
              aria-haspopup="dialog"
            >
              查看场景详情
            </a>
          </div>
        </div>
        <div className="scenario-hero-photo">
          <img src={featuredScene.image} alt={featuredScene.imageLabel} />
        </div>
      </section>

      <section className="scenario-list" aria-labelledby="scenario-list-title">
        <div className="scenario-list-heading">
          <h2 id="scenario-list-title">全部场景</h2>
          <span>已接入 {scenes.length} 个场景</span>
        </div>
        <div className="scenario-grid">
          {scenes.map((scene) => (
            <article className="scenario-card" key={scene.id}>
              <img className="scenario-card-image" src={scene.image} alt="" />
              <div className="scenario-card-copy">
                <strong>
                  <a
                    className="scenario-card-detail"
                    href={sceneDetailHref(scene.id)}
                    aria-label={`查看${scene.title}详情`}
                    aria-haspopup="dialog"
                  >
                    {scene.title}
                  </a>
                </strong>
                <span>{scene.description}</span>
              </div>
              <div className="scenario-card-action">
                <span aria-hidden="true">查看详情</span>
                <a
                  className="scenario-card-assets"
                  href={sceneAssetsHref(scene.id)}
                  aria-label={`查看${scene.title}的数据与模型`}
                >
                  数据与模型 <ArrowUpRight size={15} aria-hidden="true" />
                </a>
              </div>
            </article>
          ))}
        </div>
      </section>

      {selectedScene && (
        <SceneDetails
          key={selectedScene.id}
          scene={selectedScene}
          close={() => navigate('/scenes')}
        />
      )}
    </div>
  );
}

function sceneResources(sceneId: string, { runs, assets }: ReturnType<typeof useData>) {
  const binding = sceneBindings[sceneId];
  const runIds = new Set(binding?.runIds);
  const linkedRuns = (runs.data || []).filter((run) => runIds.has(run.id));

  const datasetIds = new Set(binding?.datasetIds);
  for (const run of linkedRuns) {
    for (const row of run.dataset_bindings || []) datasetIds.add(row.dataset_asset_id);
    for (const [role, asset] of Object.entries(run.inputs || {})) {
      if (role !== 'dataset' && !role.startsWith('dataset_')) continue;
      const id = typeof asset === 'string' ? asset : asset?.id;
      if (id) datasetIds.add(id);
    }
  }
  const linkedDatasets = (assets.data || []).filter(
    (asset) =>
      asset.kind === 'DATASET' &&
      datasetIds.has(asset.id) &&
      (asset.status || asset.state) !== 'ARCHIVED',
  );

  const modelIds = new Set(binding?.modelIds);
  const modelsFromRunOutputs = modelIds.size === 0 && runIds.size > 0;
  const availableModels = (assets.data || []).filter((asset) => {
    const state = asset.status || asset.state;
    return (
      asset.kind === 'MODEL' &&
      state !== 'ARCHIVED' &&
      state !== 'UNAVAILABLE' &&
      state !== 'MISSING' &&
      asset.metadata?.availability !== 'removed'
    );
  });
  if (modelsFromRunOutputs) {
    // Without a curated model list, show the latest available checkpoint per linked run.
    for (const run of linkedRuns) {
      const outputs = new Set(run.outputs?.map((output) => output.id));
      const latest = availableModels
        .filter((model) => outputs.has(model.id))
        .sort((left, right) => {
          const byStep = Number(right.metadata?.step ?? 0) - Number(left.metadata?.step ?? 0);
          return byStep || (right.created_at || '').localeCompare(left.created_at || '');
        })[0];
      if (latest) modelIds.add(latest.id);
    }
  }
  const linkedModels = availableModels.filter((model) => modelIds.has(model.id));

  const loadingRuns = runIds.size > 0 && !runs.data && runs.loading;
  const loadingDatasets = (datasetIds.size > 0 && !assets.data && assets.loading) || loadingRuns;
  const runMessage = loadingRuns
    ? '正在加载关联训练任务…'
    : runIds.size && runs.error
      ? '关联训练任务暂时加载失败。'
      : '暂无可查看的关联训练任务。';
  const datasetMessage = loadingDatasets
    ? '正在加载关联数据集…'
    : datasetIds.size && assets.error
      ? '关联数据集暂时加载失败。'
      : '暂无可查看的关联数据集。';
  const hasModelSources = modelIds.size > 0 || modelsFromRunOutputs;
  const loadingModels =
    (modelsFromRunOutputs && loadingRuns) || (hasModelSources && !assets.data && assets.loading);
  const modelError = (hasModelSources && assets.error) || (modelsFromRunOutputs && runs.error);
  const modelMessage = loadingModels
    ? '正在加载关联模型…'
    : modelError
      ? '关联模型暂时加载失败。'
      : '暂无可查看的关联模型。';

  return { linkedRuns, linkedDatasets, linkedModels, runMessage, datasetMessage, modelMessage };
}

export function getSceneAssetFilter(sceneId: string, data: ReturnType<typeof useData>) {
  const scene = scenes.find((item) => item.id === sceneId);
  if (!scene) return undefined;
  const { linkedDatasets, linkedModels } = sceneResources(scene.id, data);
  return {
    id: scene.id,
    title: scene.title,
    datasetIds: linkedDatasets.map((asset) => asset.id),
    modelIds: linkedModels.map((asset) => asset.id),
  };
}

function SceneDetails({ scene, close }: { scene: Scene; close: () => void }) {
  const { linkedRuns, linkedDatasets, linkedModels, runMessage, datasetMessage, modelMessage } =
    sceneResources(scene.id, useData());
  const resources = [
    {
      label: '训练任务',
      Icon: Workflow,
      message: runMessage,
      links: linkedRuns.map(({ id, name }) => ({
        id,
        name,
        href: `#/runs/${encodeURIComponent(id)}`,
      })),
    },
    {
      label: '数据集',
      Icon: Database,
      message: datasetMessage,
      links: linkedDatasets.map(({ id, name }) => ({
        id,
        name,
        href: sceneAssetsHref(scene.id, id),
      })),
    },
    {
      label: '模型',
      Icon: Box,
      message: modelMessage,
      links: linkedModels.map(({ id, name }) => ({
        id,
        name,
        href: sceneAssetsHref(scene.id, id),
      })),
    },
  ];

  return (
    <Modal title={scene.title} close={close} wide className="scenario-showcase-dialog">
      <div className="scenario-detail-lead">
        <div className="scenario-detail-photo">
          <img src={scene.image} alt={scene.imageLabel} />
        </div>
        <div>
          <Badge state="AVAILABLE" label="已接入" />
          <p>{scene.task}</p>
        </div>
      </div>

      <section aria-labelledby="scenario-related-title">
        <div className="scenario-related-heading">
          <h3 id="scenario-related-title">关联资源</h3>
          <a className="button outline small" href={sceneAssetsHref(scene.id)}>
            查看数据与模型 <ArrowRight size={14} aria-hidden="true" />
          </a>
        </div>
        <div className="scenario-resources">
          {resources.map(({ label, Icon, message, links }) => (
            <div className="scenario-resource-group" key={label}>
              <h4>
                {label}
                {links.length > 0 ? `（${links.length}）` : null}
              </h4>
              {links.length ? (
                <ul className="scenario-resource-links">
                  {links.map(({ id, name, href }) => (
                    <li key={id}>
                      <a className="scenario-resource-link" href={href}>
                        <Icon size={16} aria-hidden="true" />
                        <span>{name}</span>
                        <ArrowRight size={15} aria-hidden="true" />
                      </a>
                    </li>
                  ))}
                </ul>
              ) : (
                <p role="status">{message}</p>
              )}
            </div>
          ))}
        </div>
      </section>

      <section aria-labelledby="scenario-stages-title">
        <h3 id="scenario-stages-title">平台支持的环节</h3>
        <dl className="scenario-stages">
          <div>
            <dt>数据准备</dt>
            <dd>{scene.data}</dd>
          </div>
          <div>
            <dt>策略训练</dt>
            <dd>{scene.training}</dd>
          </div>
          <div>
            <dt>真机验证</dt>
            <dd>{scene.validation}</dd>
          </div>
        </dl>
      </section>
    </Modal>
  );
}
