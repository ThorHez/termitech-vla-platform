import type { Node } from './types';

export function nodeAddressText(node: Node): string {
  const address = node.connection_address;
  if (!address?.host) return '';
  if (address.port === null) return address.host;
  const host = address.host.includes(':') ? `[${address.host}]` : address.host;
  return `${host}:${address.port}`;
}

export function matchesNodeQuery(node: Node, query: string): boolean {
  const target = query.trim().toLowerCase();
  return [node.name, node.hostname, node.connection_address?.host || '', nodeAddressText(node)]
    .join(' ')
    .toLowerCase()
    .includes(target);
}
