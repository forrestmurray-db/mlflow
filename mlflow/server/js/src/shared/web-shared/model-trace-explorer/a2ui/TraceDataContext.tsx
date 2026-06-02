import { createContext, useContext, useMemo } from 'react';
import type { ReactNode } from 'react';

import type { ModelTraceSpanNode } from '../ModelTrace.types';

export interface SpanSelector {
  span_id?: string | null;
  name?: string | null;
  kind?: string | null;
  attributes?: Record<string, string | number | boolean> | null;
  // legacy MLflow shorthand; matches against node.type
  span_type?: string | null;
}

export interface TraceDataContextValue {
  rootNode: ModelTraceSpanNode | null;
  nodeMap: Record<string, ModelTraceSpanNode>;
  findSpansBySelector: (selector: SpanSelector) => ModelTraceSpanNode[];
}

const TraceDataContext = createContext<TraceDataContextValue | null>(null);

const flattenNodes = (node: ModelTraceSpanNode | null): ModelTraceSpanNode[] => {
  if (!node) return [];
  const out: ModelTraceSpanNode[] = [node];
  for (const child of node.children ?? []) {
    out.push(...flattenNodes(child));
  }
  return out;
};

const matchesSelector = (node: ModelTraceSpanNode, selector: SpanSelector): boolean => {
  if (selector.span_id && node.key !== selector.span_id) return false;
  if (selector.name && node.title !== selector.name) return false;
  if (selector.span_type && node.type !== selector.span_type) return false;
  if (selector.attributes) {
    const attrs = (node.attributes ?? {}) as Record<string, unknown>;
    for (const [k, v] of Object.entries(selector.attributes)) {
      if (attrs[k] !== v) return false;
    }
  }
  return true;
};

export const TraceDataProvider = ({
  rootNode,
  nodeMap,
  children,
}: {
  rootNode: ModelTraceSpanNode | null;
  nodeMap: Record<string, ModelTraceSpanNode>;
  children: ReactNode;
}) => {
  const value = useMemo<TraceDataContextValue>(() => {
    const allNodes = flattenNodes(rootNode);
    return {
      rootNode,
      nodeMap,
      findSpansBySelector: (selector: SpanSelector) => allNodes.filter((n) => matchesSelector(n, selector)),
    };
  }, [rootNode, nodeMap]);

  return <TraceDataContext.Provider value={value}>{children}</TraceDataContext.Provider>;
};

export const useTraceData = (): TraceDataContextValue => {
  const ctx = useContext(TraceDataContext);
  if (!ctx) {
    throw new Error('useTraceData must be used within a TraceDataProvider');
  }
  return ctx;
};
