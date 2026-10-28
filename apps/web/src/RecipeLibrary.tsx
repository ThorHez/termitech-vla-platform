import { useEffect, useRef, useState } from 'react';
import { BookOpen, FolderHeart, RefreshCw } from 'lucide-react';
import { useApi } from './api';
import { useAuth } from './auth';
import { Badge, ErrorNotice, Field } from './components';
import { RecipeAutocomplete } from './RecipeAutocomplete';
import { RecipeDetails } from './RecipeDetails';
import type { RecipeEditorDraft } from './trainingDraft';
import type {
  Node,
  Profile,
  RecipeDatasetSlot,
  RecipeDetail,
  RecipeRevision,
  SavedRecipe,
} from './types';

export function RecipeLibrary({
  node,
  profile,
  recipe,
  initialRevision,
  onSelect,
  onDefaults,
  onDatasets,
  onResolved,
  onDirtyChange,
  onLoadingChange,
  restoredEditorDraft,
  onEditorDraftChange,
}: {
  node?: Node;
  profile?: Profile;
  recipe: string;
  initialRevision?: RecipeRevision;
  onSelect: (recipe: string, revision?: RecipeRevision) => void;
  onDefaults: (defaults: Record<string, number>, recipe: string, revisionId?: string) => void;
  onDatasets: (slots: RecipeDatasetSlot[], recipe: string, revisionId?: string) => void;
  onResolved?: (detail: RecipeDetail, recipe: string, revisionId?: string) => void;
  onDirtyChange: (dirty: boolean) => void;
  onLoadingChange: (loading: boolean) => void;
  restoredEditorDraft?: RecipeEditorDraft;
  onEditorDraftChange?: (draft: RecipeEditorDraft | undefined) => void;
}) {
  const { can } = useAuth();
  const initial =
    initialRevision?.definition.node_id === node?.id &&
    initialRevision?.definition.profile_id === profile?.id
      ? initialRevision
      : undefined;
  const [mode, setMode] = useState(initial ? 'saved' : 'code');
  const [saveNotice, setSaveNotice] = useState('');
  const [confirmedName, setConfirmedName] = useState<string>();
  const [selected, setSelected] = useState<SavedRecipe | undefined>(
    initial
      ? {
          id: initial.recipe_id,
          name: initial.definition.name,
          description: initial.description || '',
          node_id: initial.definition.node_id,
          profile_id: initial.definition.profile_id,
          latest_revision: initial,
          created_at: initial.created_at,
          validated: initial.validated,
        }
      : undefined,
  );
  const lastCode = useRef(recipe);
  const library = useApi<SavedRecipe[]>(
    node && profile
      ? `/recipes?node_id=${encodeURIComponent(node.id)}&profile_id=${encodeURIComponent(profile.id)}`
      : null,
    10000,
  );
  const choices =
    library.data?.filter((item) => item.node_id === node?.id && item.profile_id === profile?.id) ||
    [];
  const editorScope = JSON.stringify([
    node?.id,
    profile?.id,
    profile?.code_dir,
    profile?.python,
    profile?.adapter,
    recipe,
    selected?.latest_revision.id,
  ]);
  // A matching draft may open the detail reader, but only its fresh response
  // can authorize restoring edits. Do not restore autocomplete confirmation.
  const hasEditorDraft = restoredEditorDraft?.scope === editorScope;
  useEffect(() => {
    setSelected((previous) => {
      if (!previous) return previous;
      const latest = library.data?.find(
        (item) =>
          item.id === previous.id && item.latest_revision.id === previous.latest_revision.id,
      );
      if (!latest || latest.latest_revision.validated === previous.latest_revision.validated)
        return previous;
      return {
        ...previous,
        validated: latest.validated,
        status: latest.status,
        latest_revision: {
          ...previous.latest_revision,
          validated: latest.latest_revision.validated,
          status: latest.latest_revision.status,
        },
      };
    });
  }, [library.data]);
  const selectSaved = (value: SavedRecipe) => {
    setSelected(value);
    setMode('saved');
    onDirtyChange(false);
    onSelect(value.latest_revision.definition.base_recipe, value.latest_revision);
  };
  return (
    <div className="recipe-library">
      <div className="recipe-mode-switch" role="group" aria-label="配方来源">
        <button
          type="button"
          aria-pressed={mode === 'code'}
          className={mode === 'code' ? 'selected' : ''}
          onClick={() => {
            if (mode === 'code') return;
            setMode('code');
            setSelected(undefined);
            onDirtyChange(false);
            onSelect(lastCode.current);
          }}
        >
          <BookOpen size={14} />
          代码配方
        </button>
        <button
          type="button"
          aria-pressed={mode === 'saved'}
          className={mode === 'saved' ? 'selected' : ''}
          onClick={() => {
            if (mode === 'saved') return;
            lastCode.current = recipe;
            setMode('saved');
            setSelected(undefined);
            onDirtyChange(false);
            onSelect('');
          }}
        >
          <FolderHeart size={14} />
          自定义配方{choices.length > 0 && <span>{choices.length}</span>}
        </button>
      </div>
      {mode === 'code' ? (
        can('training.write') ? (
          <RecipeAutocomplete
            value={recipe}
            onChange={(value) => {
              lastCode.current = value;
              onSelect(value);
            }}
            node={node}
            profile={profile}
            onResolvedName={setConfirmedName}
          />
        ) : (
          <p className="field-hint">代码配方读取需要操作员或管理员权限。可查看已保存的配方。</p>
        )
      ) : (
        <>
          <Field
            label="自定义配方"
            required
            hint="只显示当前机器和运行环境的配方。选择时固定版本，不随后续更新改变。"
          >
            <select
              value={selected?.latest_revision.id || ''}
              onChange={(event) => {
                const item = choices.find(
                  (entry) => entry.latest_revision.id === event.target.value,
                );
                if (item) selectSaved(item);
              }}
            >
              <option value="">
                {library.loading && !library.data
                  ? '正在读取配方库…'
                  : choices.length
                    ? '选择已保存的配方'
                    : '暂无配方，可从代码配方另存创建'}
              </option>
              {selected &&
                !choices.some(
                  (entry) => entry.latest_revision.id === selected.latest_revision.id,
                ) && (
                  <option value={selected.latest_revision.id}>
                    {selected.name} · v{selected.latest_revision.revision}（已选版本）
                  </option>
                )}
              {choices.map((item) => (
                <option key={item.latest_revision.id} value={item.latest_revision.id}>
                  {item.name} · v{item.latest_revision.revision} ·{' '}
                  {item.latest_revision.validated ? '已有验收记录' : '预检时验证'}
                </option>
              ))}
            </select>
          </Field>
          <ErrorNotice error={library.error} retry={library.refresh} />
          <button
            type="button"
            className="text-link recipe-library-refresh"
            onClick={library.refresh}
          >
            <RefreshCw size={12} />
            刷新配方库
          </button>
          {selected && (
            <div className="recipe-library-selection">
              <div>
                <strong>{selected.name}</strong>
                <span>
                  v{selected.latest_revision.revision} · 基于{' '}
                  {selected.latest_revision.definition.base_recipe}
                </span>
                {selected.description && <p>{selected.description}</p>}
              </div>
              <Badge
                state={selected.latest_revision.validated ? 'READY' : 'PLANNED'}
                label={selected.latest_revision.validated ? '已有验收记录' : '预检时验证'}
              />
            </div>
          )}
        </>
      )}
      {mode === 'saved' || (recipe && (confirmedName === recipe || hasEditorDraft)) ? (
        <RecipeDetails
          node={node}
          profile={profile}
          recipe={recipe}
          saved={selected}
          onSaved={(value) => {
            library.refresh();
            selectSaved(value);
            setSaveNotice(
              `已保存并选中 ${value.name} · v${value.latest_revision.revision}。${can('recipes.writeback') ? '配置已写入代码并提交。' : '已保存到平台配方库。'}Agent 将在预检时自动验证。`,
            );
          }}
          onDefaults={onDefaults}
          onDatasets={onDatasets}
          onResolved={onResolved}
          onDirtyChange={onDirtyChange}
          onLoadingChange={onLoadingChange}
          restoredEditorDraft={restoredEditorDraft}
          onEditorDraftChange={onEditorDraftChange}
        />
      ) : (
        <p className="field-hint">
          从联想结果选择完整配方名称后展示详细配置；输入关键字仅用于搜索。
        </p>
      )}
      {saveNotice && (
        <p className="recipe-detail-warning" role="status">
          {saveNotice}
        </p>
      )}
    </div>
  );
}
