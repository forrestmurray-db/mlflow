import { useCallback, useMemo } from 'react';

import { A2UIViewer, litTheme } from '@a2ui/react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { CreateAssessmentPayload } from '../api';
import type { FeedbackAssessment } from '../ModelTrace.types';
import { useCreateAssessment } from '../hooks/useCreateAssessment';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { buildSampleSpec } from './buildSampleSpec';
import { registerMlflowCatalog } from './catalog/registerMlflowCatalog';
import { FeedbackActionProvider } from './FeedbackActionContext';
import type { SubmitFeedbackArgs } from './FeedbackActionContext';
import { TraceDataProvider } from './TraceDataContext';

// Register the catalog with the singleton on module load so A2UIViewer sees it
// when it consults ComponentRegistry.getInstance().
registerMlflowCatalog();

export const TraceViewA2UIPrototype = () => {
  const { theme } = useDesignSystemTheme();
  const { rootNode, nodeMap } = useModelTraceExplorerViewState();
  const traceId = rootNode?.traceId ?? '';

  const { createAssessmentMutation } = useCreateAssessment({ traceId });

  const handleSubmitFeedback = useCallback(
    (args: SubmitFeedbackArgs) => {
      if (!traceId) return;
      const assessment: Omit<FeedbackAssessment, 'assessment_id' | 'create_time' | 'last_update_time'> = {
        assessment_name: args.name,
        trace_id: traceId,
        span_id: args.spanId,
        source: { source_type: 'HUMAN', source_id: 'a2ui-prototype' },
        feedback: { value: args.value },
        rationale: args.comment,
        metadata: {
          view_source: 'a2ui-prototype',
          ...(args.elementId ? { element_id: args.elementId } : {}),
        },
      };
      const payload: CreateAssessmentPayload = { assessment };
      createAssessmentMutation(payload);
    },
    [createAssessmentMutation, traceId],
  );

  const sampleSpec = useMemo(() => buildSampleSpec(), []);

  if (!rootNode) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <Typography.Text color="secondary">No trace loaded.</Typography.Text>
      </div>
    );
  }

  return (
    <TraceDataProvider rootNode={rootNode} nodeMap={nodeMap}>
      <FeedbackActionProvider onSubmit={handleSubmitFeedback}>
        <div
          css={{
            padding: theme.spacing.lg,
            overflow: 'auto',
            height: '100%',
          }}
        >
          <A2UIViewer
            root={sampleSpec.root}
            components={sampleSpec.components}
            data={{}}
            theme={litTheme}
            onAction={(action) => {
              // a2ui's own action plumbing is not used in this prototype; feedback
              // routes through React context. Logged here so we can see what a2ui
              // emits when interacting with built-in components.
              // eslint-disable-next-line no-console
              console.debug('[a2ui prototype] action', action);
            }}
          />
        </div>
      </FeedbackActionProvider>
    </TraceDataProvider>
  );
};
