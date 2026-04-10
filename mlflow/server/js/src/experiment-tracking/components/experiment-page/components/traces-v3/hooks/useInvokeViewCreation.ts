import { useMutation } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../../../common/utils/FetchUtils';

interface InvokeViewCreationParams {
  experimentId: string;
  traceIds: string[];
  provider: string;
  model: string;
  secret_id?: string;
  endpoint_name?: string;
}

interface InvokeViewCreationResponse {
  job_id: string;
  run_id: string;
}

export const useInvokeViewCreation = () => {
  return useMutation<InvokeViewCreationResponse, Error, InvokeViewCreationParams>({
    mutationFn: async (params) => {
      const response = await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/traces/views/invoke'), {
        method: 'POST',
        body: {
          experiment_id: params.experimentId,
          trace_ids: params.traceIds,
          provider: params.provider,
          model: params.model,
          secret_id: params.secret_id,
          endpoint_name: params.endpoint_name,
        },
      });
      return response as InvokeViewCreationResponse;
    },
  });
};
