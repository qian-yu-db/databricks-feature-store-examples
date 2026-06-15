# MLOps DAB Template Customization Guide

This guide walks through simplifying the [Databricks MLOps Stacks](https://github.com/databricks/mlops-stacks) template into a leaner, single-cloud DAB template tailored to your organization.

## Goals

Simplify the Mlops Stack Template Setup

- Single cloud provider (no multi-cloud conditionals)
- Single CI/CD platform
- Only 2 workspaces: staging and prod environments
- Required + optional pipelines (not all-or-nothing)
- Fewer parameters for data scientists to fill in at project creation time

---

## Step 1: Lock In Fixed Parameters

Before modifying any code, decide what is constant across all generated projects. These become hardcoded values instead of user-prompted parameters.

| Decision | Options | Impact |
|----------|---------|--------|
| Cloud provider | aws / azure / gcp | Removes `input_cloud`, simplifies node types, doc links, workspace URL defaults |
| CI/CD platform | github_actions / azure_devops / gitlab | Removes `input_cicd_platform`, deletes 2 of 3 CI/CD directories |
| Unity Catalog | yes / no | If yes: removes workspace model registry code. If no: removes UC catalog/schema parameters |
| MLflow Recipes | yes / no / optional | If no: removes `training/steps/`, `training/profiles/`, recipe notebooks |

## Step 2: Simplify `databricks_template_schema.json`

This file defines all user-facing parameters. Remove parameters that are now fixed.

**Parameters to remove (hardcode instead):**

| Parameter | Why |
|-----------|-----|
| `input_cloud` | Single cloud — hardcode everywhere |
| `input_cicd_platform` | Single platform — delete unused CI/CD dirs |
| `input_setup_cicd_and_project` | Always generate both — remove `CICD_Only` / `Project_Only` modes |
| `input_docker_image` | Only needed for GitLab — remove if not using it |
| `input_include_mlflow_recipes` | Remove if your org doesn't use MLflow Recipes |

**Parameters to potentially remove (if Unity Catalog is fixed):**

| Parameter | Condition |
|-----------|-----------|
| `input_include_models_in_unity_catalog` | Hardcode if org standardizes on UC |
| `input_staging_catalog_name` | Hardcode if catalog names are fixed |
| `input_prod_catalog_name` | Hardcode if catalog names are fixed |
| `input_test_catalog_name` | Remove — no test environment |
| `input_schema_name` | Keep or default to project name |
| `input_unity_catalog_read_user_group` | Hardcode if org has a standard group |

**Parameters to keep:**

| Parameter | Reason |
|-----------|--------|
| `input_project_name` | Always unique per project |
| `input_root_dir` | Needed for monorepo support |
| `input_databricks_staging_workspace_host` | May vary across teams |
| `input_databricks_prod_workspace_host` | May vary across teams |
| `input_default_branch` | Usually `main` but keep for flexibility |
| `input_release_branch` | Usually `release` but keep for flexibility |
| `input_read_user_group` | Varies by team |

**New parameters to add (optional pipelines):**

```json
"input_include_batch_inference": {
  "order": 10,
  "type": "string",
  "description": "Include batch inference pipeline",
  "default": "yes",
  "enum": ["yes", "no"]
},
"input_include_monitoring": {
  "order": 11,
  "type": "string",
  "description": "Include model monitoring pipeline",
  "default": "no",
  "enum": ["yes", "no"]
},
"input_include_feature_store": {
  "order": 12,
  "type": "string",
  "description": "Include feature engineering pipeline",
  "default": "no",
  "enum": ["yes", "no"]
}
```

After editing, also clean up all `skip_prompt_if` blocks that reference removed parameters.

## Step 3: Remove Unused Cloud and CI/CD Files

### 3a. Delete unused CI/CD directories

Under `template/{{.input_root_dir}}/`, keep only your chosen platform's directory:

| Keep | Delete |
|------|--------|
| `.github/` | `.azure/`, `.gitlab/` |
| `.azure/` | `.github/`, `.gitlab/` |
| `.gitlab/` | `.github/`, `.azure/` |

### 3b. Simplify `library/template_variables.tmpl`

Replace cloud conditionals with a single value:

```
# Before (multi-cloud)
{{ define `cloud_specific_node_type_id` -}}
    {{- if (eq .input_cloud `aws`) -}}
        i3.xlarge
    {{- else if (eq .input_cloud `azure`) -}}
        Standard_D3_v2
    {{- else if (eq .input_cloud `gcp`) -}}
        n2-highmem-4
    {{- end -}}
{{- end -}}

# After (single cloud, e.g. azure)
{{ define `cloud_specific_node_type_id` -}}
    Standard_D3_v2
{{- end -}}
```

Do the same for `databricks_staging_workspace_host` and `databricks_prod_workspace_host` default values.

### 3c. Simplify `library/functions.tmpl`

`generate_doc_link` has branches for all three clouds. Keep only your cloud's branch.

## Step 4: Simplify `update_layout.tmpl`

This is where most conditional file inclusion/exclusion lives.

**Remove these blocks entirely:**
- `input_setup_cicd_and_project` conditionals (always `CICD_and_Project` now)
- CI/CD platform skip blocks (only one platform)
- Cloud-specific conditionals (if any)

**Add skip blocks for optional pipelines:**

```
{{ if (eq .input_include_batch_inference `no`) }}
    {{ skip (printf `%s/%s/%s` $root_dir $project_name `deployment/batch_inference`) }}
    {{ skip (printf `%s/%s/%s` $root_dir $project_name `resources/batch-inference-workflow-resource.yml`) }}
{{ end }}

{{ if (eq .input_include_monitoring `no`) }}
    {{ skip (printf `%s/%s/%s` $root_dir $project_name `monitoring`) }}
    {{ skip (printf `%s/%s/%s` $root_dir $project_name `resources/monitoring-resource.yml`) }}
{{ end }}

{{ if (eq .input_include_feature_store `no`) }}
    {{ skip (printf `%s/%s/%s` $root_dir $project_name `feature_engineering`) }}
    {{ skip (printf `%s/%s/%s` $root_dir $project_name `tests/feature_engineering`) }}
    {{ skip (printf `%s/%s/%s` $root_dir $project_name `training/notebooks/TrainWithFeatureStore.py`) }}
    {{ skip (printf `%s/%s/%s` $root_dir $project_name `resources/feature-engineering-workflow-resource.yml`) }}
{{ end }}
```

## Step 5: Simplify Targets to Staging + Prod

Edit the generated project's `databricks.yml` template:

`template/{{.input_root_dir}}/{{template 'project_name_alphanumeric_underscore' .}}/databricks.yml`

- Remove the `dev` and `test` targets
- Keep only `staging` and `prod`
- Update any references to removed targets in CI/CD workflow templates

Also update the CI/CD workflow templates to remove dev-related deploy steps.

## Step 6: Update Tests

### 6a. Simplify `tests/utils.py`

- Remove `AWS_DEFAULT_PARAMS` / `GCP_DEFAULT_PARAMS` (or whichever don't apply)
- Keep only your cloud's defaults
- Remove `parametrize_by_cloud` — tests run against one cloud
- Simplify `parametrize_by_project_generation_params` to only cover remaining feature combos

### 6b. Update test files

- `test_create_project.py`: Remove cloud parametrization, update GCP+UC skip conditions if not applicable
- `test_github_actions.py` / `test_gitlab.py` / `test_bundle_resources.py`: Keep only the tests relevant to your CI/CD platform
- Update `tests/example-project-configs/` — keep only your cloud's configs

### 6c. Run tests after each step

```bash
pytest tests -vv --black
```

This catches breakage incrementally. The test matrix should shrink from ~99 variants to ~10 or fewer.

## Step 7: Clean Up Documentation

- Update `template/{{.input_root_dir}}/README.md.tmpl` to remove references to removed options
- Update `template/{{.input_root_dir}}/docs/` if present
- Update the root `README.md` and `stack-customization.md` to reflect your simplified template
- Remove `Pipeline.md` if the pipeline diagrams no longer match

## Recommended Execution Order

| Phase | Steps | Validation |
|-------|-------|------------|
| 1. Core simplification | Steps 1-3 | `pytest tests -vv --black` |
| 2. Layout and targets | Steps 4-5 | `pytest tests -vv --black` |
| 3. Tests and docs | Steps 6-7 | `pytest tests -vv --black` + manual `databricks bundle init` preview |

Generate a preview after each phase to visually verify the output:

```bash
databricks bundle init . --config-file <your-config>.json --output-dir /tmp/preview
```

---

## Requirements Questionnaire

Use this questionnaire to collect requirements from your ML engineering and data science teams before customizing the template.

### Infrastructure

1. **Which cloud provider does your team use for Databricks?**
   - [ ] AWS
   - [ ] Azure
   - [ ] GCP

2. **What is the URL of your staging Databricks workspace?**
   - `https://___________`

3. **What is the URL of your production Databricks workspace?**
   - `https://___________`

4. **Which CI/CD platform does your team use?**
   - [ ] GitHub Actions
   - [ ] GitHub Actions (GitHub Enterprise Server)
   - [ ] Azure DevOps
   - [ ] GitLab

5. **What is your default git branch name?** (e.g., `main`)
   - `___________`

6. **What is your release branch name?** (e.g., `release`)
   - `___________`

### Model Registry and Governance

7. **Do you use Unity Catalog for model registration?**
   - [ ] Yes
   - [ ] No (using workspace model registry)

8. **If using Unity Catalog, what are your catalog names?**
   - Staging catalog: `___________`
   - Production catalog: `___________`

9. **What schema naming convention do you prefer for ML artifacts?**
   - [ ] Use the project name as the schema name
   - [ ] Fixed schema name: `___________`

### ML Pipelines

10. **Which pipelines should be included in every new project? (required)**
    - [ ] Model training
    - [ ] Model validation
    - [ ] Batch inference
    - [ ] Model monitoring
    - [ ] Feature engineering

11. **Which pipelines should be available as opt-in options? (optional per project)**
    - [ ] Batch inference
    - [ ] Model monitoring
    - [ ] Feature engineering

12. **What training approach does your team primarily use?**
    - [ ] Notebook-based training with Delta Tables (default)
    - [ ] Feature Store integration
    - [ ] MLflow Recipes
    - [ ] Let each project choose

### Compute and Resources

13. **What is the preferred instance type for ML training jobs?**
    - (Default for your cloud will be used if not specified)
    - `___________`

14. **Do your ML jobs typically use single-node or cluster compute?**
    - [ ] Single node
    - [ ] Multi-node cluster
    - [ ] Depends on the project

### Permissions and Access

15. **What user group should have READ access to ML resources?**
    - (e.g., `users`, `data-science-team`)
    - `___________`

16. **If using Unity Catalog, what group should have EXECUTE access to registered models?**
    - (e.g., `account users`)
    - `___________`

### Project Conventions

17. **What naming convention do you use for ML projects?**
    - (e.g., `team-name-model-purpose`, `project-name`)
    - `___________`

18. **Do you use a monorepo structure (multiple projects in one repo)?**
    - [ ] Yes — root directory name: `___________`
    - [ ] No — one project per repo

19. **Any additional pipelines, jobs, or resources your team needs that aren't covered above?**
    - `___________`

20. **Any existing internal tools or patterns that should be integrated into the template?**
    - `___________`
