import React from 'react';
import { Button, SparkleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

interface CreateViewsButtonProps {
  componentId: string;
  onClick: () => void;
}

export const CreateViewsButton: React.FC<CreateViewsButtonProps> = ({ componentId, onClick }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <Button
      componentId={componentId}
      onClick={onClick}
      aria-label={intl.formatMessage({
        defaultMessage: 'Create views for traces',
        description: 'Aria label for the create views button',
      })}
      icon={<SparkleIcon color="ai" />}
      css={{
        border: '1px solid transparent !important',
        background: `linear-gradient(${theme.colors.backgroundPrimary}, ${theme.colors.backgroundPrimary}) padding-box, ${theme.gradients.aiBorderGradient} border-box`,
      }}
    >
      <FormattedMessage defaultMessage="Create Views" description="Label for the create views button" />
    </Button>
  );
};
