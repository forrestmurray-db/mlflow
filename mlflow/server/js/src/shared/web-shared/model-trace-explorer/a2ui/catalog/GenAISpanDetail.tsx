import { memo } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { A2UIComponentProps } from '@a2ui/react';

import { applyJsonPathToObject } from '../../hooks/useTraceViewFiltering';
import type { SpanSelector } from '../TraceDataContext';
import { useTraceData } from '../TraceDataContext';

interface GenAISpanDetailProperties {
  selector: SpanSelector;
  inputPath?: string | null;
  outputPath?: string | null;
  title?: string | null;
}

const renderExtractedValue = (value: unknown): string => {
  if (value == null) return '—';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

export const GenAISpanDetail = memo(function GenAISpanDetail({ node }: A2UIComponentProps) {
  const { theme } = useDesignSystemTheme();
  const { findSpansBySelector } = useTraceData();

  const properties = (node.properties ?? {}) as unknown as GenAISpanDetailProperties;
  const { selector, inputPath, outputPath, title } = properties;

  const matches = selector ? findSpansBySelector(selector) : [];
  const span = matches[0];

  const titleText = title ?? (span ? `${span.title} (${span.type ?? 'span'})` : 'GenAISpanDetail');

  const inputValue = span ? applyJsonPathToObject(span.inputs, inputPath ?? null) : undefined;
  const outputValue = span ? applyJsonPathToObject(span.outputs, outputPath ?? null) : undefined;

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.md,
        backgroundColor: theme.colors.backgroundPrimary,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
      }}
    >
      <Typography.Title level={4} withoutMargins>
        {titleText}
      </Typography.Title>

      {!span && (
        <Typography.Text color="secondary">
          {matches.length === 0 ? 'No span matched the selector.' : `${matches.length} spans matched (showing first).`}
        </Typography.Text>
      )}

      {span && (
        <>
          {inputValue != null && (
            <div>
              <Typography.Text bold size="sm" color="secondary">
                Input
              </Typography.Text>
              <pre
                css={{
                  margin: 0,
                  padding: theme.spacing.sm,
                  backgroundColor: theme.colors.backgroundSecondary,
                  borderRadius: theme.borders.borderRadiusSm,
                  fontSize: theme.typography.fontSizeSm,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: 240,
                  overflow: 'auto',
                }}
              >
                {renderExtractedValue(inputValue)}
              </pre>
            </div>
          )}

          {outputValue != null && (
            <div>
              <Typography.Text bold size="sm" color="secondary">
                Output
              </Typography.Text>
              <pre
                css={{
                  margin: 0,
                  padding: theme.spacing.sm,
                  backgroundColor: theme.colors.backgroundSecondary,
                  borderRadius: theme.borders.borderRadiusSm,
                  fontSize: theme.typography.fontSizeSm,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: 240,
                  overflow: 'auto',
                }}
              >
                {renderExtractedValue(outputValue)}
              </pre>
            </div>
          )}
        </>
      )}
    </div>
  );
});
