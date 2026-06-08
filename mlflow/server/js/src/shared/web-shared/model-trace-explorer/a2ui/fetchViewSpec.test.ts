import { fetchTraceViewSpec } from './fetchViewSpec';
import { fetchAPI, getAjaxUrl } from '../ModelTraceExplorer.request.utils';

jest.mock('../ModelTraceExplorer.request.utils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: jest.fn((url: string) => url),
}));

describe('fetchTraceViewSpec', () => {
  it('POSTs trace_id and model to the view-spec endpoint and returns the spec', async () => {
    const spec = { name: 'V', root: 'col-root', components: [] };
    (fetchAPI as jest.Mock).mockResolvedValue(spec);

    const result = await fetchTraceViewSpec({ traceId: 'tr-1', model: 'openai:/gpt-4o' });

    expect(getAjaxUrl).toHaveBeenCalledWith('ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec');
    expect(fetchAPI).toHaveBeenCalledWith(
      'ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec',
      'POST',
      { trace_id: 'tr-1', model: 'openai:/gpt-4o' },
    );
    expect(result).toBe(spec);
  });
});
