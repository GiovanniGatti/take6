import os

from azureml import core, exceptions


def get_workspace() -> core.Workspace:
    try:
        current_run = core.Run.get_context(allow_offline=False)
    except exceptions.RunEnvironmentException:
        # Running locally, use cli authentication
        subscription_id = os.environ['AZUREML_ARM_SUBSCRIPTION']
        resource_group = os.environ['AZUREML_ARM_RESOURCEGROUP']
        workspace_name = os.environ['AZUREML_ARM_WORKSPACE_NAME']
        return core.Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)
    return current_run.experiment.workspace
