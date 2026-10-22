"""Identical recipe-name matching and ordering on the Agent and control plane."""


def recipe_matches(name: str, query: str) -> bool:
    return query.casefold() in name.casefold()


def recipe_sort_key(name: str, query: str) -> tuple:
    folded = name.casefold()
    return (not folded.startswith(query.casefold()), folded, name)
