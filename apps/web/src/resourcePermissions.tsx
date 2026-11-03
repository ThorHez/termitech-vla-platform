import type { User } from './auth';
import { useAuth } from './auth';
import { useMembers } from './data';

export interface MemberSummary {
  id: string;
  username: string;
  display_name: string;
}

/** Missing historical ownership is deliberately reserved for administrators. */
export function canManageResource(
  user: Pick<User, 'id' | 'role'> | undefined,
  ownerUserId?: string | null,
) {
  return Boolean(
    user &&
      (user.role === 'admin' ||
        (user.role === 'operator' && ownerUserId && ownerUserId === user.id)),
  );
}

export function resourceOwnerLabel(
  ownerUserId: string | null | undefined,
  members: readonly MemberSummary[] = [],
  currentUser?: Pick<User, 'id' | 'username' | 'display_name'>,
) {
  if (!ownerUserId) return '历史记录（未分配创建者）';
  const owner =
    members.find((member) => member.id === ownerUserId) ||
    (currentUser?.id === ownerUserId ? currentUser : undefined);
  return owner ? owner.display_name || owner.username : `用户 ${ownerUserId.slice(0, 8)}`;
}

export function ResourceOwner({ ownerUserId }: { ownerUserId?: string | null }) {
  const { user } = useAuth();
  const members = useMembers();
  return (
    <span className="subtext">创建者：{resourceOwnerLabel(ownerUserId, members?.data, user)}</span>
  );
}
