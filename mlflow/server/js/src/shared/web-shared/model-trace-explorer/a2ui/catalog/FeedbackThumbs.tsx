import { memo, useState } from 'react';

import { Button, ThumbsDownIcon, ThumbsUpIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { A2UIComponentProps } from '@a2ui/react';

import type { SpanSelector } from '../TraceDataContext';
import { useTraceData } from '../TraceDataContext';
import { useSubmitFeedback } from '../FeedbackActionContext';

interface FeedbackThumbsProperties {
  name: string;
  target?: 'view' | 'trace' | 'span';
  spanSelector?: SpanSelector;
  label?: string | null;
}

export const FeedbackThumbs = memo(function FeedbackThumbs({ node }: A2UIComponentProps) {
  const { theme } = useDesignSystemTheme();
  const { findSpansBySelector } = useTraceData();
  const submit = useSubmitFeedback();

  const properties = (node.properties ?? {}) as unknown as FeedbackThumbsProperties;
  const { name, target = 'trace', spanSelector, label } = properties;

  const [selectedValue, setSelectedValue] = useState<boolean | null>(null);

  const handleClick = (value: boolean) => {
    setSelectedValue(value);
    const spanMatches = spanSelector ? findSpansBySelector(spanSelector) : [];
    const spanId = target === 'span' ? spanMatches[0]?.key?.toString() : undefined;
    submit({
      name,
      value,
      target,
      spanId,
      elementId: node.id,
    });
  };

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        padding: theme.spacing.sm,
      }}
    >
      <Typography.Text size="sm" color="secondary">
        {label ?? name}
      </Typography.Text>
      <Button
        componentId={`a2ui.feedback-thumbs.${node.id}.up`}
        type={selectedValue === true ? 'primary' : 'tertiary'}
        size="small"
        icon={<ThumbsUpIcon />}
        onClick={() => handleClick(true)}
        aria-label={`Thumbs up for ${name}`}
      />
      <Button
        componentId={`a2ui.feedback-thumbs.${node.id}.down`}
        type={selectedValue === false ? 'primary' : 'tertiary'}
        size="small"
        icon={<ThumbsDownIcon />}
        onClick={() => handleClick(false)}
        aria-label={`Thumbs down for ${name}`}
      />
    </div>
  );
});
