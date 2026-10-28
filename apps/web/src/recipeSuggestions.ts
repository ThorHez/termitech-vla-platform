import type { Node, Profile, RecipeSuggestion, RecipeSuggestions } from './types';

export type RecipeSuggestionNamespace = 'recipe-suggestions' | 'inference-config-suggestions';

export function recipeSuggestionScope(
  node?: Node,
  profile?: Profile,
  namespace: RecipeSuggestionNamespace = 'recipe-suggestions',
) {
  return JSON.stringify([
    namespace,
    node?.id,
    profile?.id,
    profile?.node_id,
    profile?.code_dir,
    profile?.python,
    profile?.adapter,
    profile?.execution_mode,
    profile?.recipe,
    profile?.metadata?.validated,
    profile?.metadata?.validated_recipes,
  ]);
}

export function currentRecipeEntries(
  response: RecipeSuggestions | undefined,
  node: Node | undefined,
  profile: Profile | undefined,
  value: string,
  codeOnly = false,
): RecipeSuggestion[] | undefined {
  if (
    !response ||
    response.status !== 'READY' ||
    response.node_id !== node?.id ||
    response.profile_id !== profile?.id ||
    response.prefix !== value ||
    profile?.node_id !== node?.id
  )
    return undefined;
  return codeOnly
    ? response.entries.filter((entry) => entry.source === 'code' || entry.source === 'both')
    : response.entries;
}
