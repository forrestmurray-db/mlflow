import React, { useState, useCallback, useRef } from 'react';
import {
  Modal,
  Button,
  useDesignSystemTheme,
  SparkleIcon,
  Typography,
  Alert,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import { IssueDetectionModelSelection, type IssueDetectionModelSelectionRef } from './IssueDetectionModelSelection';
import { useInvokeViewCreation } from './hooks/useInvokeViewCreation';

interface ViewCreationModalProps {
  traceIds: string[];
  experimentId: string;
  onClose: () => void;
  onSubmitSuccess?: (runId: string) => void;
}

export const ViewCreationModal: React.FC<ViewCreationModalProps> = ({
  traceIds,
  experimentId,
  onClose,
  onSubmitSuccess,
}) => {
  const { theme } = useDesignSystemTheme();
  const modelSelectionRef = useRef<IssueDetectionModelSelectionRef>(null);

  const [currentStep, setCurrentStep] = useState<1 | 2>(1);
  const [isModelSelectionValid, setIsModelSelectionValid] = useState(false);

  const {
    mutate: createSecret,
    isLoading: isCreatingSecret,
    error: createSecretError,
    reset: resetCreateSecret,
  } = useCreateSecret();

  const {
    mutate: invokeViewCreation,
    isLoading: isInvokingViewCreation,
    error: viewCreationError,
    reset: resetViewCreation,
  } = useInvokeViewCreation();

  const resetForm = useCallback(() => {
    setCurrentStep(1);
    setIsModelSelectionValid(false);
    modelSelectionRef.current?.reset();
  }, []);

  const handleNext = useCallback(() => {
    setCurrentStep(2);
  }, []);

  const handlePrevious = useCallback(() => {
    setCurrentStep(1);
  }, []);

  const handleSubmit = () => {
    const values = modelSelectionRef.current?.getValues();
    if (!values) return;

    const { mode, endpointName, provider, model, apiKeyConfig, saveKey } = values;

    const submitViewCreation = (secretId?: string) => {
      invokeViewCreation(
        {
          experimentId,
          traceIds,
          provider,
          model,
          secret_id: secretId,
          endpoint_name: endpointName,
        },
        {
          onSuccess: (response) => {
            onSubmitSuccess?.(response.run_id);
            resetForm();
            onClose();
          },
        },
      );
    };

    // Endpoint mode - use the selected endpoint
    if (mode === 'endpoint' && endpointName) {
      submitViewCreation();
      return;
    }

    // Direct mode - save secret if new API key, or use existing secret
    if (mode === 'direct' && saveKey && apiKeyConfig.mode === 'new') {
      const authConfig = { ...apiKeyConfig.newSecret.configFields } satisfies Record<string, string>;
      if (apiKeyConfig.newSecret.authMode) {
        authConfig['auth_mode'] = apiKeyConfig.newSecret.authMode;
      }

      createSecret(
        {
          secret_name: apiKeyConfig.newSecret.name,
          secret_value: apiKeyConfig.newSecret.secretFields,
          provider: provider,
          auth_config: Object.keys(authConfig).length > 0 ? authConfig : undefined,
        },
        {
          onSuccess: (response) => {
            submitViewCreation(response.secret.secret_id);
          },
        },
      );
    } else if (apiKeyConfig.mode === 'existing' && apiKeyConfig.existingSecretId) {
      submitViewCreation(apiKeyConfig.existingSecretId);
    }
  };

  const handleClose = useCallback(() => {
    resetForm();
    resetCreateSecret();
    resetViewCreation();
    onClose();
  }, [resetForm, resetCreateSecret, resetViewCreation, onClose]);

  const isStep2Valid = isModelSelectionValid;

  const handleModelSelectionValidityChange = useCallback((isValid: boolean) => {
    setIsModelSelectionValid(isValid);
  }, []);

  const renderStep1Footer = () => (
    <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
      <Button componentId="mlflow.traces.view-creation-modal.cancel" onClick={handleClose}>
        <FormattedMessage defaultMessage="Cancel" description="Cancel button in view creation modal" />
      </Button>
      <Button
        componentId="mlflow.traces.view-creation-modal.next"
        type="primary"
        onClick={handleNext}
        endIcon={<ChevronRightIcon />}
      >
        <FormattedMessage defaultMessage="Next" description="Next button to proceed to provider configuration" />
      </Button>
    </div>
  );

  const renderStep2Footer = () => (
    <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
      <Button
        componentId="mlflow.traces.view-creation-modal.previous"
        onClick={handlePrevious}
        icon={<ChevronLeftIcon />}
      >
        <FormattedMessage defaultMessage="Previous" description="Previous button to go back to confirmation" />
      </Button>
      <Button
        componentId="mlflow.traces.view-creation-modal.submit"
        type="primary"
        onClick={handleSubmit}
        loading={isCreatingSecret || isInvokingViewCreation}
        disabled={!isStep2Valid}
      >
        <SparkleIcon css={{ marginRight: theme.spacing.xs }} />
        <FormattedMessage defaultMessage="Create Views" description="Submit button to trigger view creation job" />
      </Button>
    </div>
  );

  return (
    <Modal
      componentId="mlflow.traces.view-creation-modal"
      title={
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <SparkleIcon color="ai" />
          <FormattedMessage
            defaultMessage="Create Trace Views"
            description="Title of the trace view creation modal"
          />
        </div>
      }
      visible
      onCancel={handleClose}
      footer={currentStep === 1 ? renderStep1Footer() : renderStep2Footer()}
    >
      {(createSecretError || viewCreationError) && (
        <Alert
          componentId="mlflow.traces.view-creation-modal.error"
          type="error"
          message={createSecretError?.message || viewCreationError?.message}
          closable
          onClose={() => {
            resetCreateSecret();
            resetViewCreation();
          }}
          css={{ marginBottom: theme.spacing.md }}
        />
      )}
      {currentStep === 1 ? (
        <>
          <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.lg }}>
            <FormattedMessage
              defaultMessage="This will analyze {count, plural, one {1 trace} other {# traces}} and create milestone views for each."
              description="Confirmation message showing how many traces will be analyzed"
              values={{ count: traceIds.length }}
            />
          </Typography.Text>
        </>
      ) : (
        <IssueDetectionModelSelection
          ref={modelSelectionRef}
          selectedTraceIds={traceIds}
          onSelectTracesClick={() => {}}
          onValidityChange={handleModelSelectionValidityChange}
        />
      )}
    </Modal>
  );
};
