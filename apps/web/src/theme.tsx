import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { Monitor, Moon, Sun } from 'lucide-react';

export type ThemePreference = 'light' | 'dark' | 'system';
const storageKey = 'termitech.theme';
const mediaQuery = '(prefers-color-scheme: dark)';
const options = [
  { value: 'light', label: '浅色模式', icon: Sun },
  { value: 'dark', label: '夜间模式', icon: Moon },
  { value: 'system', label: '跟随系统', icon: Monitor },
] as const;

export function parseThemePreference(value: string | null): ThemePreference {
  return value === 'light' || value === 'dark' ? value : 'system';
}

function readPreference(): ThemePreference {
  try {
    return parseThemePreference(window.localStorage.getItem(storageKey));
  } catch {
    return 'system';
  }
}

const ThemeContext = createContext<{
  preference: ThemePreference;
  setPreference: (preference: ThemePreference) => void;
}>({ preference: 'system', setPreference: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, updatePreference] = useState<ThemePreference>(readPreference);
  const [systemDark, setSystemDark] = useState(
    () => typeof window !== 'undefined' && !!window.matchMedia?.(mediaQuery).matches,
  );

  useEffect(() => {
    const media = window.matchMedia?.(mediaQuery);
    if (!media) return;
    const update = () => setSystemDark(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    const sync = (event: StorageEvent) => {
      if (event.key === storageKey || event.key === null) {
        updatePreference(readPreference());
      }
    };
    window.addEventListener('storage', sync);
    return () => window.removeEventListener('storage', sync);
  }, []);

  useEffect(() => {
    const theme = preference === 'system' ? (systemDark ? 'dark' : 'light') : preference;
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute('content', theme === 'dark' ? '#0d1620' : '#f5f7fa');
  }, [preference, systemDark]);

  const setPreference = (next: ThemePreference) => {
    updatePreference(next);
    try {
      window.localStorage.setItem(storageKey, next);
    } catch {
      // Keep the current tab usable even if storage is blocked or full.
    }
  };

  return (
    <ThemeContext.Provider value={{ preference, setPreference }}>{children}</ThemeContext.Provider>
  );
}

export function ThemeControl() {
  const { preference, setPreference } = useContext(ThemeContext);
  const Icon = options.find((option) => option.value === preference)!.icon;
  return (
    <label className="theme-control" title="切换界面主题">
      <Icon size={16} aria-hidden="true" />
      <select
        aria-label="界面主题"
        value={preference}
        onChange={(event) => setPreference(event.target.value as ThemePreference)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ThemeSettings() {
  const { preference, setPreference } = useContext(ThemeContext);
  return (
    <div className="theme-settings">
      <div className="theme-options" role="group" aria-label="外观主题">
        {options.map(({ value, label, icon: Icon }) => (
          <button
            type="button"
            className={`theme-option ${value === preference ? 'selected' : ''}`}
            key={value}
            aria-pressed={value === preference}
            onClick={() => setPreference(value)}
          >
            <Icon size={20} aria-hidden="true" />
            {label}
          </button>
        ))}
      </div>
      <p>主题偏好保存在当前浏览器；选择“跟随系统”时，会随系统外观自动切换。</p>
    </div>
  );
}
