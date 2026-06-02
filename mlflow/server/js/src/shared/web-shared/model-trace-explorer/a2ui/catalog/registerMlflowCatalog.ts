import { ComponentRegistry, initializeDefaultCatalog } from '@a2ui/react';

import { FeedbackThumbs } from './FeedbackThumbs';
import { GenAISpanDetail } from './GenAISpanDetail';

let registered = false;

export const registerMlflowCatalog = () => {
  if (registered) return;
  registered = true;

  initializeDefaultCatalog();

  const registry = ComponentRegistry.getInstance();
  registry.register('GenAISpanDetail', { component: GenAISpanDetail });
  registry.register('FeedbackThumbs', { component: FeedbackThumbs });
};
