# Campfire Publishing

The [publication](https://github.com/gsk-tech/channel_agnostic_vit/tree/publication) branch of this repository will be mirrored at [GSK-AI/campfire] for public consumption. The workflow to publish this exists at `.github/workflows/publish.yml`.

## Pre-Reqs

Running the publication workflow requires access to GitHub App credentials owned by the GSK-AI GitHub organization, which is installed on the `GSK-AI/campfire` repository. This GitHub App has permission scopes `contents:write` allowing it to publish to the `main` branch of that public repo. These values are persisted as Actions Secrets in this repository. Any need to change the secrets will require an admin of the GSK-AI organization to re-provision, then re-populate in this repository.

|Secret Name|Description|
|-|-|
|GH_APP_CLIENT_ID| The Client ID of the campfire-publish GitHub App.|
|GH_APP_PRIVATE_KEY|The Private Key of the campfire-publish GitHub App.|

## Workflow

> [!IMPORTANT]  
> A change to the git commit history can disrupt this workflow.

The workflow will checkout this repository on a push to the `publication` branch, generate a GitHub App token, update the git config to point to the public repository, then push the entire commit tree to that repository.
