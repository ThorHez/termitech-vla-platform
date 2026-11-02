export interface DatasetBindingDisclosureState {
  count: number;
  expanded: number[];
}

type DisclosureAction =
  | { type: 'toggle' | 'open' | 'remove'; index: number }
  | { type: 'append' }
  | { type: 'sync'; count: number };

export function createDatasetBindingDisclosure(count: number): DatasetBindingDisclosureState {
  return { count, expanded: count > 0 ? [count - 1] : [] };
}

export function datasetBindingDisclosureReducer(
  state: DatasetBindingDisclosureState,
  action: DisclosureAction,
): DatasetBindingDisclosureState {
  if (action.type === 'append') return { count: state.count + 1, expanded: [state.count] };
  if (action.type === 'sync') {
    if (action.count === state.count) return state;
    if (action.count > state.count) return createDatasetBindingDisclosure(action.count);
    return {
      count: action.count,
      expanded: [
        ...new Set(state.expanded.map((index) => Math.min(index, action.count - 1))),
      ].filter((index) => index >= 0),
    };
  }
  if (action.index < 0 || action.index >= state.count) return state;
  if (action.type === 'remove') {
    const count = state.count - 1;
    const expanded = state.expanded
      .filter((index) => index !== action.index)
      .map((index) => (index > action.index ? index - 1 : index));
    if (!expanded.length && count > 0 && state.expanded.includes(action.index))
      expanded.push(Math.min(action.index, count - 1));
    return { count, expanded };
  }
  const expanded = state.expanded.includes(action.index)
    ? action.type === 'open'
      ? state.expanded
      : state.expanded.filter((index) => index !== action.index)
    : [...state.expanded, action.index];
  return expanded === state.expanded ? state : { ...state, expanded };
}
