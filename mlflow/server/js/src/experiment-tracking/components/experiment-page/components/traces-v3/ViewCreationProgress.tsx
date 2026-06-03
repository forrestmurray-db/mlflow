import { FormattedMessage, useIntl } from 'react-intl';
import {
  Button,
  CheckCircleIcon,
  ParagraphSkeleton,
  XCircleIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import Utils from '../../../../../common/utils/Utils';
import { JobStatus, isJobComplete } from '../../../../components/run-page/hooks/useFetchJobStatus';
import { useCancelJob } from '../../../../components/run-page/hooks/useCancelJob';

export interface ViewCreationJobResult {
  views_created?: number;
  errors?: number;
  total_traces_analyzed?: number;
}

export interface ViewCreationProgressProps {
  /** Job ID for cancel functionality */
  jobId?: string;
  /** Job status from parent */
  jobStatus?: JobStatus;
  /** Current job stage */
  jobStage?: string;
  /** Total traces count */
  totalTraces?: number;
  /** Job result data */
  result?: ViewCreationJobResult;
  /** Whether job status is loading */
  isLoadingJobStatus?: boolean;
  /** Error from job status fetch */
  jobStatusError?: Error | null;
  /** Error message from backend when job failed */
  jobErrorMessage?: string;
}

export const ViewCreationProgress = ({
  jobId,
  jobStatus,
  jobStage,
  totalTraces,
  result,
  isLoadingJobStatus,
  jobStatusError,
  jobErrorMessage,
}: ViewCreationProgressProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { cancelJob, isCancelling } = useCancelJob();

  const handleCancel = () => {
    if (!jobId) return;
    cancelJob(
      { jobId },
      {
        onError: (error) => {
          Utils.logErrorAndNotifyUser(
            intl.formatMessage(
              {
                defaultMessage: 'Failed to cancel job: {error}',
                description: 'Error message when job cancellation fails',
              },
              { error: error.message },
            ),
          );
        },
      },
    );
  };

  const isJobSucceeded = jobStatus === JobStatus.SUCCEEDED;
  const isJobFailed = jobStatus === JobStatus.FAILED || jobStatus === JobStatus.TIMEOUT || !!jobStatusError;
  const isJobCanceled = jobStatus === JobStatus.CANCELED;
  const jobComplete = isJobComplete(jobStatus) || !!jobStatusError;

  const viewsCreated = result?.views_created ?? 0;
  const errorCount = result?.errors ?? 0;

  if (isLoadingJobStatus) {
    return (
      <div css={{ marginBottom: theme.spacing.lg }}>
        <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
          <FormattedMessage defaultMessage="View creation progress" description="View creation progress > Title" />
        </Typography.Title>
        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: theme.spacing.md,
          }}
        >
          <ParagraphSkeleton />
        </div>
      </div>
    );
  }

  return (
    <div css={{ marginBottom: theme.spacing.lg }}>
      <div
        css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.sm }}
      >
        <Typography.Title level={4} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="View creation progress" description="View creation progress > Title" />
        </Typography.Title>
        {!jobComplete && (
          <Button componentId="mlflow.traces.view-creation.cancel-button" onClick={handleCancel} loading={isCancelling}>
            <FormattedMessage defaultMessage="Cancel" description="View creation progress > Cancel button" />
          </Button>
        )}
      </div>

      <div
        css={{
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          padding: theme.spacing.md,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.xs }}>
          {isJobSucceeded ? (
            <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />
          ) : isJobFailed || isJobCanceled ? (
            <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />
          ) : (
            <div
              css={{
                width: 16,
                height: 16,
                borderRadius: '50%',
                border: `2px solid ${theme.colors.border}`,
                borderTopColor: theme.colors.actionPrimaryBackgroundDefault,
                animation: 'spin 1s linear infinite',
                '@keyframes spin': {
                  '0%': { transform: 'rotate(0deg)' },
                  '100%': { transform: 'rotate(360deg)' },
                },
              }}
            />
          )}
          <Typography.Text>
            {isJobSucceeded ? (
              <FormattedMessage
                defaultMessage="View creation completed"
                description="View creation progress > Step label when completed"
              />
            ) : isJobFailed ? (
              <FormattedMessage
                defaultMessage="View creation failed"
                description="View creation progress > Step label when failed"
              />
            ) : isJobCanceled ? (
              <FormattedMessage
                defaultMessage="View creation canceled"
                description="View creation progress > Step label when canceled"
              />
            ) : jobStage ? (
              jobStage
            ) : (
              <FormattedMessage
                defaultMessage="Creating views from traces..."
                description="View creation progress > Step label (fallback)"
              />
            )}
          </Typography.Text>
        </div>
        <div css={{ marginLeft: 24 }}>
          <Typography.Hint>
            {jobComplete ? (
              errorCount > 0 ? (
                <FormattedMessage
                  defaultMessage="{viewsCreated} {viewsCreated, plural, one {view} other {views}} created for {totalTraces} {totalTraces, plural, one {trace} other {traces}} ({errorCount} {errorCount, plural, one {error} other {errors}})"
                  description="View creation progress > Progress summary when completed with errors"
                  values={{ viewsCreated, totalTraces, errorCount }}
                />
              ) : (
                <FormattedMessage
                  defaultMessage="{viewsCreated} {viewsCreated, plural, one {view} other {views}} created for {totalTraces} {totalTraces, plural, one {trace} other {traces}}"
                  description="View creation progress > Progress summary when completed"
                  values={{ viewsCreated, totalTraces }}
                />
              )
            ) : (
              <FormattedMessage
                defaultMessage="Creating views for {totalTraces} {totalTraces, plural, one {trace} other {traces}}, {viewsCreated} {viewsCreated, plural, one {view} other {views}} created so far"
                description="View creation progress > Progress summary while running"
                values={{ viewsCreated, totalTraces }}
              />
            )}
          </Typography.Hint>
        </div>
      </div>

      {isJobFailed && jobErrorMessage && (
        <>
          <Typography.Title level={4} css={{ marginTop: theme.spacing.lg, marginBottom: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Error details" description="View creation error details > Title" />
          </Typography.Title>
          <div
            css={{
              border: `1px solid ${theme.colors.actionDangerPrimaryBackgroundDefault}`,
              borderRadius: theme.borders.borderRadiusMd,
              padding: theme.spacing.md,
              backgroundColor: `${theme.colors.actionDangerPrimaryBackgroundDefault}15`,
            }}
          >
            <Typography.Text css={{ color: theme.colors.textValidationDanger, wordBreak: 'break-word' }}>
              {jobErrorMessage}
            </Typography.Text>
          </div>
        </>
      )}
    </div>
  );
};
