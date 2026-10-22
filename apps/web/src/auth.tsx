import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  ApiError,
  api,
  discardLegacyToken,
  post,
  registerReauthentication,
  setSessionCsrf,
  subscribeSessionExpiry,
} from './api';
import { ReauthenticationDialog } from './ReauthenticationDialog';
import { watchHumanActivity } from './sessionActivity';

export type UserRole = 'admin' | 'operator' | 'viewer';
export interface User {
  id: string;
  username: string;
  display_name: string;
  role: UserRole;
  enabled: boolean;
  must_change_password: boolean;
  created_at: string;
  updated_at: string;
}
export interface AuthSession {
  user: User;
  permissions: string[];
  csrf_token: string;
  expires_at: string;
  idle_expires_at: string;
  session_idle_seconds: number;
  reauth_expires_at: string | null;
  server_time: string;
}
export const roleLabels: Record<UserRole, string> = {
  admin: '管理员',
  operator: '操作员',
  viewer: '只读用户',
};
export const roleDescriptions: Record<UserRole, string> = {
  admin: '管理用户、机器连接、资产与训练，并可回写配方和查看审计。',
  operator: '查看资源，管理资产与配方，创建和取消训练。',
  viewer: '查看训练、资产与资源状态。',
};
interface AuthState {
  user?: User;
  permissions: string[];
  expiresAt?: string;
  idleExpiresAt?: string;
  loading: boolean;
  error?: Error;
  expired: boolean;
  can: (permission: string) => boolean;
  refresh: () => Promise<void>;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
}
const unavailable = async () => {
  throw new Error('请先登录管理服务。');
};
export const AuthContext = createContext<AuthState>({
  permissions: [],
  loading: false,
  expired: false,
  can: () => false,
  refresh: unavailable,
  login: unavailable,
  logout: unavailable,
  changePassword: unavailable,
});
export const useAuth = () => useContext(AuthContext);

export function sessionRemainingTime(session: AuthSession) {
  const serverNow = Date.parse(session.server_time);
  return {
    absolute: Date.parse(session.expires_at) - serverNow,
    idle: Date.parse(session.idle_expires_at) - serverNow,
  };
}

interface SessionChannel {
  publish: () => void;
  publishActivity: () => void;
  close: () => void;
}

// Share only an invalidation signal. Cookies, CSRF values and account data stay
// in their own browser contexts and are read again from the server.
export function openSessionChannel(
  onChange: () => void,
  onActivity?: () => void,
): SessionChannel | undefined {
  if (typeof BroadcastChannel === 'undefined') return undefined;
  try {
    const channel = new BroadcastChannel('termitech:human-session');
    channel.onmessage = (event: MessageEvent<unknown>) => {
      if (event.data === 'session-changed') onChange();
      else if (event.data === 'session-activity') onActivity?.();
    };
    return {
      publish: () => channel.postMessage('session-changed'),
      publishActivity: () => channel.postMessage('session-activity'),
      close: () => channel.close(),
    };
  } catch {
    return undefined;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<AuthSession>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error>();
  const [expired, setExpired] = useState(false);
  const [checkingDeadline, setCheckingDeadline] = useState(false);
  const [reauthenticationOpen, setReauthenticationOpen] = useState(false);
  const revision = useRef(0);
  const hadSession = useRef(false);
  const sessionRef = useRef<AuthSession | undefined>(undefined);
  const deadlines = useRef<{ absolute: number; idle: number } | undefined>(undefined);
  const sessionChannel = useRef<SessionChannel | undefined>(undefined);
  const reauthentication = useRef<
    | {
        promise: Promise<void>;
        resolve: () => void;
        reject: (cause: Error) => void;
        waiters: number;
      }
    | undefined
  >(undefined);
  const cancelReauthentication = useCallback((message = '已取消验证，操作尚未执行。') => {
    const pending = reauthentication.current;
    reauthentication.current = undefined;
    setReauthenticationOpen(false);
    pending?.reject(new ApiError('REAUTH_CANCELLED', message, 403));
  }, []);
  const clearIdentity = useCallback(
    (isExpired: boolean, cause?: Error) => {
      revision.current += 1;
      setSessionCsrf();
      sessionRef.current = undefined;
      deadlines.current = undefined;
      hadSession.current = false;
      setSession(undefined);
      setExpired(isExpired);
      setError(cause);
      setLoading(false);
      setCheckingDeadline(false);
      cancelReauthentication('登录状态已改变，操作尚未执行。');
    },
    [cancelReauthentication],
  );
  const accept = useCallback(
    (next: AuthSession) => {
      const previous = sessionRef.current;
      const remaining = sessionRemainingTime(next);
      const now = performance.now();
      const absolute = now + remaining.absolute;
      const idle = now + remaining.idle;
      const sameSession = previous?.csrf_token === next.csrf_token;
      deadlines.current = {
        absolute:
          sameSession && deadlines.current
            ? Math.min(deadlines.current.absolute, absolute)
            : absolute,
        idle: sameSession && deadlines.current ? Math.max(deadlines.current.idle, idle) : idle,
      };
      if (previous && previous.csrf_token !== next.csrf_token) {
        cancelReauthentication('登录状态已改变，请重新执行操作。');
      }
      // Concurrent polling may return an older snapshot of the same session.
      if (
        previous?.csrf_token === next.csrf_token &&
        Date.parse(previous.idle_expires_at) > Date.parse(next.idle_expires_at)
      ) {
        next = { ...next, idle_expires_at: previous.idle_expires_at };
      }
      revision.current += 1;
      hadSession.current = true;
      sessionRef.current = next;
      setSessionCsrf(next.csrf_token);
      setSession(next);
      setExpired(false);
      setError(undefined);
      setLoading(false);
      setCheckingDeadline(false);
    },
    [cancelReauthentication],
  );
  const refresh = useCallback(async () => {
    const current = revision.current;
    try {
      const next = await api<AuthSession>('/auth/me');
      if (current === revision.current) accept(next);
    } catch (cause) {
      if (current !== revision.current) return;
      if (cause instanceof ApiError && cause.status === 401) {
        clearIdentity(hadSession.current);
      } else setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      if (current === revision.current) setLoading(false);
    }
  }, [accept, clearIdentity]);
  useEffect(() => {
    discardLegacyToken();
    const unsubscribe = subscribeSessionExpiry(() => {
      clearIdentity(true);
    });
    const invalidateAndRefresh = () => {
      // Invalidate first so cached pages and privileged controls disappear even
      // while the replacement account request is still waiting on the network.
      clearIdentity(false);
      setLoading(true);
      void refresh();
    };
    const channel = openSessionChannel(invalidateAndRefresh, () => void refresh());
    sessionChannel.current = channel;
    if (!channel) window.addEventListener('focus', invalidateAndRefresh);
    const onVisibility = () => {
      if (document.visibilityState === 'visible') void refresh();
    };
    document.addEventListener('visibilitychange', onVisibility);
    void refresh();
    const timer = setInterval(() => {
      void refresh();
    }, 60000);
    return () => {
      unsubscribe();
      channel?.close();
      if (sessionChannel.current === channel) sessionChannel.current = undefined;
      if (!channel) window.removeEventListener('focus', invalidateAndRefresh);
      document.removeEventListener('visibilitychange', onVisibility);
      clearInterval(timer);
      cancelReauthentication();
    };
  }, [refresh, clearIdentity, cancelReauthentication]);
  useEffect(() => {
    if (!session?.csrf_token) return;
    return watchHumanActivity(() => {
      const current = sessionRef.current;
      if (!current) return;
      void post<AuthSession>('/auth/activity')
        .then((next) => {
          if (sessionRef.current?.csrf_token !== current.csrf_token) return;
          accept(next);
          sessionChannel.current?.publishActivity();
        })
        .catch(() => {
          // A 401 is handled globally. A network failure cannot extend the local
          // deadline; another real interaction can retry the heartbeat.
        });
    });
  }, [session?.csrf_token, accept]);
  useEffect(() => {
    if (!session) return;
    const deadline = Math.min(deadlines.current!.absolute, deadlines.current!.idle);
    const timer = setTimeout(
      () => {
        const current = revision.current;
        // A sibling tab may have extended the server's idle deadline. Hide the
        // workspace until that is checked, without extending the session ourselves.
        setCheckingDeadline(true);
        void api<AuthSession>('/auth/me')
          .then((next) => {
            if (current === revision.current) accept(next);
          })
          .catch((cause) => {
            if (current !== revision.current) return;
            clearIdentity(
              true,
              cause instanceof ApiError && cause.status === 401
                ? undefined
                : cause instanceof Error
                  ? cause
                  : new Error(String(cause)),
            );
          });
      },
      Math.max(1000, deadline - performance.now()),
    );
    return () => clearTimeout(timer);
  }, [session, accept, clearIdentity]);
  useEffect(
    () =>
      registerReauthentication((signal) => {
        if (!sessionRef.current) {
          return Promise.reject(new ApiError('UNAUTHORIZED', '请先重新登录。', 401));
        }
        signal?.throwIfAborted();
        if (!reauthentication.current) {
          let resolve!: () => void;
          let reject!: (cause: Error) => void;
          const promise = new Promise<void>((accept, decline) => {
            resolve = accept;
            reject = decline;
          });
          reauthentication.current = { promise, resolve, reject, waiters: 0 };
          setReauthenticationOpen(true);
        }
        const pending = reauthentication.current;
        pending.waiters += 1;
        return new Promise<void>((resolve, reject) => {
          let settled = false;
          const cleanup = () => {
            if (settled) return false;
            settled = true;
            pending.waiters -= 1;
            signal?.removeEventListener('abort', onAbort);
            return true;
          };
          const onAbort = () => {
            if (!cleanup()) return;
            reject(signal?.reason || new DOMException('操作已取消。', 'AbortError'));
            if (pending.waiters === 0 && reauthentication.current === pending) {
              cancelReauthentication();
            }
          };
          signal?.addEventListener('abort', onAbort, { once: true });
          pending.promise.then(
            () => {
              if (cleanup()) resolve();
            },
            (cause) => {
              if (cleanup()) reject(cause);
            },
          );
        });
      }),
    [cancelReauthentication],
  );
  const confirmReauthentication = async (password: string) => {
    const current = sessionRef.current;
    const pending = reauthentication.current;
    if (!current || !pending) return;
    const next = await post<AuthSession>('/auth/reauth', { password });
    if (
      reauthentication.current !== pending ||
      sessionRef.current?.csrf_token !== current.csrf_token
    ) {
      return;
    }
    accept(next);
    reauthentication.current = undefined;
    setReauthenticationOpen(false);
    pending.resolve();
  };
  const login = async (username: string, password: string) => {
    accept(await post<AuthSession>('/auth/login', { username, password }));
    sessionChannel.current?.publish();
  };
  const logout = async () => {
    // A failed request must leave an error visible; it has not revoked the cookie.
    await post('/auth/logout');
    clearIdentity(false);
    sessionChannel.current?.publish();
  };
  const changePassword = async (currentPassword: string, newPassword: string) => {
    accept(
      await post<AuthSession>('/auth/password', {
        current_password: currentPassword,
        new_password: newPassword,
      }),
    );
    sessionChannel.current?.publish();
  };
  return (
    <AuthContext.Provider
      value={{
        user: checkingDeadline ? undefined : session?.user,
        permissions: checkingDeadline ? [] : session?.permissions || [],
        expiresAt: session?.expires_at,
        idleExpiresAt: session?.idle_expires_at,
        loading: loading || checkingDeadline,
        error,
        expired,
        can: (permission) =>
          !!session &&
          !checkingDeadline &&
          !session.user.must_change_password &&
          session.permissions.includes(permission),
        refresh,
        login,
        logout,
        changePassword,
      }}
    >
      {children}
      {reauthenticationOpen && session && !checkingDeadline && (
        <ReauthenticationDialog
          username={session.user.username}
          confirm={confirmReauthentication}
          cancel={() => cancelReauthentication()}
        />
      )}
    </AuthContext.Provider>
  );
}
