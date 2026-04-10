/**
 * Service layer for Trace Analysis Agent API calls.
 * Handles SSE streaming from the agent server for trace analysis mode.
 */

import { getAjaxUrl, getDefaultHeaders } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { SendMessageStreamCallbacks, SendMessageStreamResult } from './AssistantService';

const TRACE_ANALYSIS_URL = getAjaxUrl('ajax-api/3.0/mlflow/assistant/trace-analysis/message');

/**
 * Send a trace analysis message and stream the SSE response.
 * POSTs to the agent server with stream=true, then reads the response body as a stream.
 * Parses `data: {json}\n\n` lines, calling callbacks for content, errors, and completion.
 * Returns an object with a close method for cancellation.
 */
export const sendTraceAnalysisStream = async (
  messages: Array<{ role: string; content: string }>,
  context: { traceId?: string; experimentId?: string },
  callbacks: SendMessageStreamCallbacks,
): Promise<SendMessageStreamResult> => {
  const { onMessage, onError, onDone } = callbacks;

  const controller = new AbortController();

  try {
    // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
    const response = await fetch(TRACE_ANALYSIS_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getDefaultHeaders(document.cookie),
      },
      body: JSON.stringify({
        messages,
        context: {
          trace_id: context.traceId,
          experiment_id: context.experimentId,
        },
        stream: true,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const error = await response.text();
      onError(`Failed to send message: ${error}`);
      return { eventSource: null };
    }

    if (!response.body) {
      onError('No response body from server');
      return { eventSource: null };
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const processBuffer = () => {
      // Split on double newlines to get SSE events
      const events = buffer.split('\n\n');
      // Keep the last (potentially incomplete) chunk in the buffer
      buffer = events.pop() ?? '';

      for (const event of events) {
        const line = event.trim();
        if (!line.startsWith('data:')) {
          continue;
        }

        const dataStr = line.slice('data:'.length).trim();

        if (dataStr === '[DONE]') {
          onDone();
          return;
        }

        try {
          const data = JSON.parse(dataStr);
          if (data.error) {
            onError(data.error);
          } else if (typeof data.content === 'string') {
            onMessage(data.content);
          } else if (typeof data.text === 'string') {
            onMessage(data.text);
          }
        } catch {
          // Ignore unparseable chunks
        }
      }
    };

    const pump = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            // Process any remaining buffered data
            if (buffer.trim()) {
              processBuffer();
            }
            onDone();
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          processBuffer();
        }
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          onError(err instanceof Error ? err.message : 'Stream read error');
        }
      }
    };

    pump();

    return {
      eventSource: null,
      close: () => controller.abort(),
    } as SendMessageStreamResult & { close: () => void };
  } catch (error) {
    if ((error as Error).name !== 'AbortError') {
      onError(error instanceof Error ? error.message : 'Unknown error');
    }
    return { eventSource: null };
  }
};
