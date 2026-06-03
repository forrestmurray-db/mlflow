export const shouldBlockLargeTraceDisplay = () => {
  return false;
};

// controls the size (in bytes) of a trace that is considered too large
// to display. default to 1gb for a safe limit to always display traces
export const getLargeTraceDisplaySizeThreshold = () => {
  return 1e9;
};

/**
 * Determines if traces V4 API should be used to fetch traces
 */
export const shouldUseTracesV4API = () => {
  return false;
};

/**
 * Determines if the new labeling schemas UI in trace assessments pane is enabled.
 * This feature allows users to configure feedback schemas at the experiment level
 * for labeling traces in the Traces tab.
 */
export const shouldEnableTracesTabLabelingSchemas = () => {
  return false;
};

/**
 * Determines if assessments/scores should be shown in experiment chat sessions.
 */
export const shouldEnableAssessmentsInSessions = () => {
  return true;
};

/**
 * Determines if assessments/scores should be shown in experiment chat sessions.
 */
export const shouldEnableTracesTableStatePersistence = () => {
  return false;
};

/**
 * A centralized setting enabling the new drawer UI for model trace explorer across the platform.
 */
export const shouldUseModelTraceExplorerDrawerUI = () => {
  return true;
};

export const shouldUseUnifiedModelTraceComparisonUI = () => {
  if (!shouldUseModelTraceExplorerDrawerUI()) {
    return false;
  }
  return true;
};

/**
 * Determines if running scorers from trace details drawer is enabled
 */
export const isEvaluatingTracesInDetailsViewEnabled = () => {
  return true;
};

/**
 * Prototype gate for the a2ui-based trace view renderer. Toggle by appending
 * `?a2ui=1` to the URL (either as a top-level query string or inside the
 * hash-router query — both work). The flag intentionally lives in the URL so
 * it can be enabled per-tab without rebuilding, until the rearchitecture
 * lands behind a real config flag.
 */
export const shouldEnableA2UITraceViews = () => {
  if (typeof window === 'undefined') return false;
  try {
    const fromSearch = new URLSearchParams(window.location.search).get('a2ui');
    if (fromSearch === '1') return true;
    const hashQueryIndex = window.location.hash.indexOf('?');
    if (hashQueryIndex >= 0) {
      const hashQuery = window.location.hash.slice(hashQueryIndex + 1);
      if (new URLSearchParams(hashQuery).get('a2ui') === '1') return true;
    }
    return false;
  } catch {
    return false;
  }
};
