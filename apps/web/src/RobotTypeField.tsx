import { useEffect, useId, useRef, useState } from 'react';
import { Check } from 'lucide-react';
import './robot-type-field.css';

// Suggestions stay complete even when metadata supplies a nonempty default.
const choices = [
  'ARX5',
  'UMI',
  'ALOHA',
  'Franka',
  'UR5',
  'Piper',
  'SO-100',
  'SO-101',
  '仿真机器人',
];

export function RobotTypeField({
  value,
  onChange,
  disabled = false,
  label = '机器人类型',
  hint = '默认值可直接修改；展开可选择其他类型，也可输入自定义机器人型号。',
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  label?: string;
  hint?: string;
}) {
  const id = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const expanded = open && !disabled;
  const selected = choices.findIndex(
    (choice) => choice.toLowerCase() === value.trim().toLowerCase(),
  );

  useEffect(() => {
    const closeOutside = (event: MouseEvent) => {
      if (rootRef.current && !event.composedPath().includes(rootRef.current)) setOpen(false);
    };
    document.addEventListener('click', closeOutside);
    return () => document.removeEventListener('click', closeOutside);
  }, []);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (expanded && active >= 0)
      document.getElementById(`${id}-option-${active}`)?.scrollIntoView({ block: 'nearest' });
  }, [active, expanded, id]);

  const showChoices = () => {
    setActive(selected);
    setOpen(true);
  };
  const choose = (choice: string) => {
    onChange(choice);
    inputRef.current?.focus();
    setOpen(false);
    setActive(-1);
  };

  return (
    <div className="field robot-type-field" ref={rootRef}>
      <label className="field-label" htmlFor={id}>
        {label}
        <span className="required"> *</span>
      </label>
      <div className="robot-type-control">
        <input
          id={id}
          ref={inputRef}
          required
          maxLength={100}
          disabled={disabled}
          autoComplete="off"
          placeholder="例如 ARX5、UMI、ALOHA"
          value={value}
          role="combobox"
          aria-autocomplete="none"
          aria-expanded={expanded}
          aria-controls={expanded ? `${id}-options` : undefined}
          aria-activedescendant={expanded && active >= 0 ? `${id}-option-${active}` : undefined}
          aria-describedby={`${id}-hint`}
          onFocus={showChoices}
          onClick={showChoices}
          onChange={(event) => {
            onChange(event.target.value);
            setActive(-1);
            setOpen(true);
          }}
          onKeyDown={(event) => {
            if (event.nativeEvent.isComposing) return;
            if (event.key === 'Escape' && expanded) {
              event.preventDefault();
              event.stopPropagation();
              setOpen(false);
            } else if (event.key === 'Tab') {
              setOpen(false);
            } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
              event.preventDefault();
              setOpen(true);
              setActive((previous) =>
                event.key === 'ArrowDown'
                  ? (previous + 1) % choices.length
                  : (previous <= 0 ? choices.length : previous) - 1,
              );
            } else if (event.key === 'Enter' && expanded) {
              event.preventDefault();
              if (active >= 0) choose(choices[active]);
              else setOpen(false);
            }
          }}
        />
        <button
          type="button"
          className="robot-type-toggle"
          tabIndex={-1}
          disabled={disabled}
          aria-label={`${expanded ? '收起' : '展开'}${label}选项`}
          aria-expanded={expanded}
          aria-controls={expanded ? `${id}-options` : undefined}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => {
            const nextOpen = !expanded;
            inputRef.current?.focus();
            setActive(selected);
            setOpen(nextOpen);
          }}
        />
      </div>
      {expanded && (
        <div className="path-suggestion-panel">
          <div
            id={`${id}-options`}
            role="listbox"
            aria-label={`${label}选项`}
            className="path-suggestion-list"
          >
            {choices.map((choice, index) => (
              <button
                type="button"
                role="option"
                id={`${id}-option-${index}`}
                aria-selected={selected === index}
                tabIndex={-1}
                key={choice}
                className={`path-suggestion-option ${active === index ? 'active' : ''}`}
                onMouseDown={(event) => event.preventDefault()}
                onMouseMove={() => setActive(index)}
                onClick={() => choose(choice)}
              >
                <Check
                  size={14}
                  aria-hidden="true"
                  style={{ visibility: selected === index ? 'visible' : 'hidden' }}
                />
                <span>{choice}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      <span id={`${id}-hint`} className="field-hint">
        {hint}
      </span>
    </div>
  );
}
