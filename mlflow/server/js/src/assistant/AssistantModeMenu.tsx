import { Button, DropdownMenu, PlusIcon, useDesignSystemTheme } from '@databricks/design-system';

import type { AssistantMode } from './types';

interface AssistantModeMenuProps {
  currentMode: AssistantMode;
  onModeChange: (mode: AssistantMode) => void;
}

/**
 * A + button that opens a dropdown menu for switching the assistant chat mode.
 * Currently supports switching to "Trace Analysis" mode.
 */
export const AssistantModeMenu = ({ currentMode, onModeChange }: AssistantModeMenuProps) => {
  const { theme } = useDesignSystemTheme();

  const handleTraceAnalysis = () => {
    onModeChange('trace_analysis');
  };

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId="mlflow.assistant.mode_menu.trigger"
          size="small"
          icon={<PlusIcon />}
          aria-label="Add mode"
          css={{ flexShrink: 0, marginRight: theme.spacing.xs }}
        />
      </DropdownMenu.Trigger>
      <DropdownMenu.Content minWidth={160} side="top" align="start">
        <DropdownMenu.Item
          componentId="mlflow.assistant.mode_menu.trace_analysis"
          onClick={handleTraceAnalysis}
          disabled={currentMode === 'trace_analysis'}
        >
          Trace Analysis
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
