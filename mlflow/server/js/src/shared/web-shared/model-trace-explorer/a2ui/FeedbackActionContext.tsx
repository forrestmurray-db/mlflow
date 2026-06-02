import { createContext, useContext } from 'react';
import type { ReactNode } from 'react';

export interface SubmitFeedbackArgs {
  name: string;
  value: boolean | number | string;
  target: 'view' | 'trace' | 'span';
  spanId?: string;
  comment?: string;
  elementId?: string;
}

export type SubmitFeedbackFn = (args: SubmitFeedbackArgs) => void;

const FeedbackActionContext = createContext<SubmitFeedbackFn | null>(null);

export const FeedbackActionProvider = ({ onSubmit, children }: { onSubmit: SubmitFeedbackFn; children: ReactNode }) => (
  <FeedbackActionContext.Provider value={onSubmit}>{children}</FeedbackActionContext.Provider>
);

export const useSubmitFeedback = (): SubmitFeedbackFn => {
  const fn = useContext(FeedbackActionContext);
  if (!fn) {
    throw new Error('useSubmitFeedback must be used within a FeedbackActionProvider');
  }
  return fn;
};
