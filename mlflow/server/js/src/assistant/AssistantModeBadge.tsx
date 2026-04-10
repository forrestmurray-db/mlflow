import { Tag } from '@databricks/design-system';

import type { AssistantMode } from './types';

interface AssistantModeBadgeProps {
  mode: AssistantMode;
  onClear: () => void;
}

export const AssistantModeBadge = ({ mode, onClear }: AssistantModeBadgeProps) => {
  if (mode === 'assistant') return null;

  return (
    <Tag componentId="mlflow.assistant.mode-badge" closable onClose={onClear}>
      Trace Analysis
    </Tag>
  );
};
