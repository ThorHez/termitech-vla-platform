import { useEffect, useId, useRef, useState } from 'react';
import { Check } from 'lucide-react';
import './editable-choice-field.css';

// Keep the full choice list available even when the field has a saved value.
export function EditableChoiceField({
  choices,
  value,
  onChange,
  disabled = false,
  label,
  hint,
  placeholder,
  maxLength,
  invalid,
  caseSensitive = true,
}: {
  choices: readonly string[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  label: string;
  hint: string;
  placeholder: string;
  maxLength?: number;
  invalid?: boolean;
  caseSensitive?: boolean;
}) {
  const id = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const hasChoices = choices.length > 0;
  const expanded = open && !disabled && hasChoices;
  const selected = caseSensitive
    ? choices.indexOf(value.trim())
    : choices.findIndex((choice) => choice.toLowerCase() === value.trim().toLowerCase());

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
    <div className="field editable-choice-field" ref={rootRef}>
      <label className="field-label" htmlFor={id}>
        {label}
        <span className="required"> *</span>
      </label>
      <div className="editable-choice-control">
        <input
          id={id}
          ref={inputRef}
          required
          maxLength={maxLength}
          disabled={disabled}
          autoComplete="off"
          placeholder={placeholder}
          value={value}
          role={hasChoices ? 'combobox' : undefined}
          aria-autocomplete={hasChoices ? 'none' : undefined}
          aria-expanded={hasChoices ? expanded : undefined}
          aria-invalid={invalid || undefined}
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
            } else if (hasChoices && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
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
        {hasChoices && (
          <button
            type="button"
            className="editable-choice-toggle"
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
        )}
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
