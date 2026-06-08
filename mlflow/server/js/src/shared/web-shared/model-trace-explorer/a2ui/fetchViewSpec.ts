import type { ComponentInstance } from '@a2ui/react';

import { fetchAPI, getAjaxUrl } from '../ModelTraceExplorer.request.utils';

export interface ViewSpec {
  name: string;
  root: string;
  components: ComponentInstance[];
}

const VIEW_SPEC_URL = 'ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec';

export const fetchTraceViewSpec = ({
  traceId,
  model,
}: {
  traceId: string;
  model?: string;
}): Promise<ViewSpec> =>
  fetchAPI(getAjaxUrl(VIEW_SPEC_URL), 'POST', { trace_id: traceId, model });
