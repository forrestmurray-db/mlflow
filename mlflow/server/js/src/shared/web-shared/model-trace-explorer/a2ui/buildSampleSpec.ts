import type { ComponentInstance } from '@a2ui/react';

export interface SampleSpec {
  root: string;
  components: ComponentInstance[];
}

/**
 * Returns a static a2ui document that demonstrates the v1 trace-views catalog:
 * - Column layout (a2ui standard)
 * - Text headers (a2ui standard)
 * - GenAISpanDetail bound by OTel-shorthand span_type (MLflow catalog extension)
 * - FeedbackThumbs at trace target (MLflow catalog extension)
 *
 * The spec is intentionally template-shaped: the GenAISpanDetail elements
 * tolerate missing matches so the same document works against any trace.
 */
export const buildSampleSpec = (): SampleSpec => ({
  root: 'col-root',
  components: [
    {
      id: 'col-root',
      component: {
        Column: { children: { explicitList: ['header', 'detail-llm', 'detail-tool', 'feedback-overall'] } },
      },
    },
    {
      id: 'header',
      component: {
        Text: { text: { literal: 'SME labeling — trace overview (a2ui prototype)' }, usageHint: 'h3' },
      },
    },
    {
      id: 'detail-llm',
      component: {
        GenAISpanDetail: {
          selector: { span_type: 'LLM' },
          title: 'LLM call',
        },
      },
    },
    {
      id: 'detail-tool',
      component: {
        GenAISpanDetail: {
          selector: { span_type: 'TOOL' },
          title: 'Tool call',
        },
      },
    },
    {
      id: 'feedback-overall',
      component: {
        FeedbackThumbs: {
          name: 'overall_quality',
          target: 'trace',
          label: 'Overall trace quality',
        },
      },
    },
  ],
});
