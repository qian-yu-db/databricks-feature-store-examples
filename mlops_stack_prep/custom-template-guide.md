# Guide: Creating a Custom Databricks Bundle Template

This guide explains how Databricks Asset Bundle templates work and how this custom template was created from the upstream [mlops-stacks](https://github.com/databrickslabs/mlops-stacks) template.

## Table of Contents

- [How Bundle Templates Work](#how-bundle-templates-work)
- [Simple vs MLOps Template Complexity](#simple-vs-mlops-template-complexity)
- [Template Anatomy](#template-anatomy)
- [Execution Flow](#execution-flow)
- [Dependency Map](#dependency-map)
- [How This Custom Template Was Created](#how-this-custom-template-was-created)
- [Using This Template](#using-this-template)
- [Creating Your Own Custom Template](#creating-your-own-custom-template)

---

## How Bundle Templates Work

When you run `databricks bundle init <template-path>`, the Databricks CLI:

1. Reads `databricks_template_schema.json` to determine what to ask the user
2. Collects user inputs (or uses defaults for hidden parameters)
3. Loads shared helpers from `library/*.tmpl`
4. Runs `template/run_validations.tmpl` to validate inputs
5. Renders all files under `template/` using Go template syntax
6. Runs `template/update_layout.tmpl` to remove files that don't apply
7. Outputs the final project to disk

All template files use **Go template syntax** (not Jinja2). Files ending in `.tmpl` are processed and have the extension stripped. Non-`.tmpl` files are copied as-is.

---

## Simple vs MLOps Template Complexity

The [Databricks docs on custom bundle templates](https://docs.databricks.com/aws/en/dev-tools/bundles/templates#custom-bundle-templates) show a minimal template structure. This project is significantly more complex — not because the templating system is different, but because of **what it generates**.

### A Simple Bundle Template

A basic custom template needs only two things:

```
basic-bundle-template/
├── databricks_template_schema.json    ← a few input fields
└── template/
    └── {{.project_name}}/
        ├── databricks.yml.tmpl        ← one bundle config
        ├── resources/
        └── src/
```

Schema defines prompts, template folder gets rendered. This is enough for a single job or pipeline.

### Why This MLOps Template Is More Complex

This template generates an **entire production MLOps project** with multiple conditional code paths. Here's what drives each piece of additional complexity:

**1. Multiple mutually exclusive ML code paths**

The template source ships ALL three ML patterns:
- Delta Tables (`Train.py`)
- Feature Store (`TrainWithFeatureStore.py` + `feature_engineering/`)
- MLflow Recipes (`TrainWithMLflowRecipes.py` + `training/steps/` + `training/profiles/`)

Only one is kept per generated project. This is why **`update_layout.tmpl`** exists — it uses `{{ skip }}` to remove files for the two unused paths. A simple template that generates one thing doesn't need this.

**2. Multiple cloud providers (upstream)**

The upstream template supports AWS/Azure/GCP, each with different:
- VM node types (`Standard_D3_v2` vs `i3.xlarge` vs `n2-highmem-4`)
- Documentation URLs (`learn.microsoft.com` vs `docs.aws.amazon.com`)
- Workspace URL patterns

This is why **`library/template_variables.tmpl`** and **`library/functions.tmpl`** exist — they centralize cloud-specific logic so individual `.tmpl` files don't repeat conditionals everywhere.

**3. Multiple CI/CD platforms (upstream)**

Three complete CI/CD directory trees (`.github/`, `.azure/`, `.gitlab/`) are included in the source. Only one is kept based on user selection. Again, **`update_layout.tmpl`** prunes the others.

**4. Multi-environment bundle config**

The generated project has dev/staging/prod targets, each pointing to different workspaces, catalogs, and permission groups. That's a lot of parameterization in `databricks.yml.tmpl` alone.

**5. Input validation**

With ~20 parameters and cross-field dependencies (e.g., GCP + Unity Catalog was unsupported), **`input_validation.tmpl`** existed to catch invalid combinations before rendering.

### Side-by-Side Comparison

| Aspect | Simple Template | This MLOps Template |
|--------|----------------|-------------------|
| Inputs | 2-3 fields | 20 fields (5 prompted in custom version) |
| `library/` | Optional | Needed to avoid repeating cloud/doc logic |
| `update_layout.tmpl` | Not needed | Required to prune mutually exclusive paths |
| `input_validation.tmpl` | Not needed | Validates cross-field constraints |
| Output | One job/pipeline | Full ML project (training, feature eng, deployment, validation, monitoring, resources, CI/CD) |

> **Key takeaway:** The templating system is identical. The complexity comes from generating a production MLOps project with multiple conditional code paths, not from the template engine itself.

---

## Template Anatomy

```
my-template/
├── databricks_template_schema.json   ← 1. SCHEMA: defines user prompts
├── library/                          ← 2. LIBRARY: shared helpers & variables
│   ├── template_variables.tmpl          (reusable Go template definitions)
│   ├── functions.tmpl                   (reusable helper functions)
│   └── input_validation.tmpl            (input validation rules)
├── template/                         ← 3. TEMPLATE: the actual output files
│   ├── run_validations.tmpl             (runs first, validates inputs)
│   ├── update_layout.tmpl               (runs last, removes unwanted files)
│   └── {{.input_root_dir}}/            (output directory tree)
│       └── ...all project files...
└── tests/                            ← 4. TESTS: validate generated output
```

### Component Details

### 1. `databricks_template_schema.json` — Input Schema

This is the entry point. It defines every parameter the template accepts.

```json
{
  "properties": {
    "input_project_name": {
      "order": 1,
      "type": "string",
      "default": "my_mlops_project",
      "description": "Project Name",
      "pattern": "^[^ .\\\\/]{3,}$",
      "pattern_match_failure_message": "Must be 3+ chars, no spaces or dots."
    }
  }
}
```

Key fields:

| Field | Purpose |
|-------|---------|
| `order` | Controls the sequence parameters are prompted |
| `type` | Data type (`string`) |
| `default` | Pre-filled value shown to user |
| `enum` | Restricts to a list of allowed values |
| `pattern` | Regex validation on user input |
| `pattern_match_failure_message` | Error shown when pattern fails |
| `skip_prompt_if` | Condition to hide this parameter (user is never asked; default is used) |
| `description` | Prompt text shown to user |

### 2. `library/*.tmpl` — Shared Helpers

These files define **named Go templates** available to all other template files. They are never rendered directly.

**`template_variables.tmpl`** — Derived values computed from inputs:

```go
{{ define `databricks_staging_workspace_host` -}}
    {{- with url .input_databricks_staging_workspace_host -}}
        {{ print .Scheme "://" .Host }}
    {{- end -}}
{{- end }}

{{ define `project_name_alphanumeric_underscore` -}}
    {{- (regexp `-`).ReplaceAllString
        ((regexp `[^A-Za-z0-9_-]`).ReplaceAllString .input_project_name ``) `_` -}}
{{- end }}
```

**`functions.tmpl`** — Reusable helper functions:

```go
{{ define "generate_doc_link" -}}
    https://learn.microsoft.com/azure/databricks/{{ .path }}
{{- end }}
```

**`input_validation.tmpl`** — Validation logic called before rendering.

### 3. `template/` — Output Files

Everything under `template/{{.input_root_dir}}/` becomes the generated project.

- **`.tmpl` files** are processed through Go templates, then the `.tmpl` extension is stripped
- **Non-`.tmpl` files** (images, `.gitignore`, etc.) are copied verbatim
- **Directory names** can contain template expressions: `{{template "project_name_alphanumeric_underscore" .}}`

Inside `.tmpl` files, you reference inputs and library templates:

```go
host: {{template `databricks_staging_workspace_host` .}}
catalog: {{ .input_staging_catalog_name }}
```

**`run_validations.tmpl`** runs before file rendering to validate inputs.

**`update_layout.tmpl`** runs after all files are rendered. It uses `{{ skip "path" }}` to delete files that don't apply to the user's configuration:

```go
{{ skip (printf "%s/%s" $root_dir ".github") }}
{{ skip (printf "%s/%s" $root_dir ".azure") }}
```

> The template source contains ALL possible files for every configuration.
> `update_layout.tmpl` prunes it down to just what applies.

---

## Execution Flow

```
databricks bundle init <template>
        │
        ▼
┌─────────────────────────────────┐
│  Read databricks_template_      │
│  schema.json                    │
│  → Prompt user for inputs       │
│  → Apply defaults for hidden    │
│    params (skip_prompt_if)      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Load library/*.tmpl            │
│  → template_variables.tmpl      │
│  → functions.tmpl               │
│  → input_validation.tmpl        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Execute run_validations.tmpl   │
│  → Calls {{ template            │
│    `validation` . }}            │
│  → Aborts if validation fails   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Render template/               │
│  {{.input_root_dir}}/           │
│  → Process .tmpl files          │
│  → Copy non-.tmpl files         │
│  → Evaluate directory names     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Execute update_layout.tmpl     │
│  → {{ skip }} removes files     │
│    that don't apply             │
│  → Cleans up template infra     │
│    files (itself, validations)  │
└──────────────┬──────────────────┘
               │
               ▼
        Generated project
        written to disk
```

---

## Dependency Map

This shows what references what across the template:

```
databricks_template_schema.json
  └─ defines → input_project_name, input_cloud, input_staging_catalog_name, etc.
       │
       ▼
library/template_variables.tmpl
  └─ reads inputs → defines derived templates:
       • databricks_staging_workspace_host  (cleaned URL)
       • databricks_prod_workspace_host     (cleaned URL)
       • project_name_alphanumeric_underscore (sanitized name)
       • cloud_specific_node_type_id        (Azure VM size)
       • model_name, experiment_base_name
       │
library/functions.tmpl
  └─ defines reusable helpers:
       • get_host()           (extract scheme + host from URL)
       • generate_doc_link()  (Azure-specific documentation URLs)
       │
       ▼
template/**/*.tmpl  (all project files)
  └─ uses inputs directly:     {{ .input_staging_catalog_name }}
  └─ uses library templates:   {{template `databricks_staging_workspace_host` .}}
  └─ uses library functions:   {{template "generate_doc_link" (map "path" "...")}}
       │
       ▼
template/update_layout.tmpl
  └─ reads inputs → decides which rendered files to {{ skip }}
```

---

## How This Custom Template Was Created

The upstream `mlops-stacks` template supports ~20 parameters across AWS/Azure/GCP, three CI/CD platforms, and three ML code paths. This custom template was created by:

### 1. Simplifying `databricks_template_schema.json`

The original ~20 parameters were reduced to **5 prompted parameters**. The remaining ~15 were hidden using `skip_prompt_if` with hardcoded defaults:

| Parameter | Hardcoded Value | Reason |
|-----------|----------------|--------|
| `input_cloud` | `azure` | Azure only |
| `input_include_models_in_unity_catalog` | `yes` | Always UC |
| `input_include_feature_store` | `yes` | Always use Feature Store |
| `input_include_mlflow_recipes` | `no` | Not using Recipes |
| `input_cicd_platform` | `github_actions` | Doc links only |
| `input_staging_catalog_name` | `edsvoya_p_stg` | Our staging catalog |
| `input_prod_catalog_name` | `edsvoya_prod` | Our prod catalog |
| `input_schema_name` | `edsvoya_operations` | Our schema |
| `input_default_branch` | `main` | Standard |
| `input_release_branch` | `feature` | Our convention |

**5 parameters still prompted:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_project_name` | `my_mlops_project` | Name of the ML project |
| `input_root_dir` | (same as project name) | Output directory name |
| `input_databricks_staging_workspace_host` | Pre-filled staging URL | Staging workspace |
| `input_databricks_prod_workspace_host` | Pre-filled prod URL | Production workspace |
| `input_read_user_group` | `users` | READ permissions group |

### 2. Simplifying `library/template_variables.tmpl`

Removed cloud-specific conditionals. Instead of:

```go
{{ if eq .input_cloud "azure" }}Standard_D3_v2{{ else if eq .input_cloud "aws" }}i3.xlarge{{ end }}
```

Hardcoded to:

```go
{{ define `cloud_specific_node_type_id` -}}
    Standard_D3_v2
{{- end }}
```

### 3. Simplifying `library/functions.tmpl`

`generate_doc_link` always produces Azure docs URLs instead of branching by cloud.

### 4. Hardening `template/update_layout.tmpl`

Always skips:

- **All CI/CD directories** (`.github/`, `.azure/`, `.gitlab/`) — using external Jenkins/XLDeploy
- **MLflow Recipes files** — not using this ML code path
- **Plain `Train.py`** — using Feature Store's `TrainWithFeatureStore.py` instead
- **Workspace model registry utils** — always Unity Catalog

### 5. Emptying `library/input_validation.tmpl`

Since most inputs are hardcoded, complex cross-field validation was removed.

---

## Using This Template

### Generate a New Project (Interactive)

```bash
cd ~/workspace
databricks bundle init ~/workspace/mlops-stacks-custom-template
```

You'll be prompted for 5 parameters. The output is created at `<root_dir>/<project_name>/`.

### Generate a New Project (Non-Interactive)

Create a config file (`config.json`):

```json
{
  "input_project_name": "my_ml_project",
  "input_root_dir": "my_ml_project",
  "input_databricks_staging_workspace_host": "https://adb-1111111111111111.1.azuredatabricks.net",
  "input_databricks_prod_workspace_host": "https://adb-2222222222222222.2.azuredatabricks.net",
  "input_read_user_group": "users"
}
```

```bash
databricks bundle init ~/workspace/mlops-stacks-custom-template --config-file config.json
```

### Preview Without Writing (Dry Run)

```bash
databricks bundle init ~/workspace/mlops-stacks-custom-template --output-dir /tmp/preview
```

### After Generation

```bash
cd my_ml_project/my_ml_project
databricks bundle validate        # check bundle config
databricks bundle deploy -t dev   # deploy to dev target
```

---

## Creating Your Own Custom Template

If you want to create a similar custom template for a different environment:

### Step 1: Fork the Upstream Template

```bash
git clone https://github.com/databrickslabs/mlops-stacks.git my-custom-template
cd my-custom-template
```

### Step 2: Identify What to Hardcode

Decide which parameters should be fixed for your organization:

- Cloud provider (aws / azure / gcp)
- Unity Catalog vs. workspace model registry
- Feature Store vs. Delta Tables vs. MLflow Recipes
- CI/CD platform
- Catalog and schema names
- Workspace URLs

### Step 3: Edit `databricks_template_schema.json`

For each parameter you want to hide:

1. Set `default` to your chosen value
2. Add `enum` with only that value
3. Add `skip_prompt_if` to suppress the prompt:

```json
"input_cloud": {
  "order": 101,
  "type": "string",
  "default": "azure",
  "enum": ["azure"],
  "skip_prompt_if": {
    "properties": {
      "input_project_name": { "pattern": ".*" }
    }
  }
}
```

> The `skip_prompt_if` pattern `".*"` matches anything — so this parameter is always skipped.

### Step 4: Simplify `library/template_variables.tmpl`

Remove cloud-specific conditionals and replace with your hardcoded values:

```go
// Before (multi-cloud):
{{ define `cloud_specific_node_type_id` -}}
    {{- if eq .input_cloud "azure" -}} Standard_D3_v2
    {{- else if eq .input_cloud "aws" -}} i3.xlarge
    {{- else -}} n2-highmem-4
    {{- end -}}
{{- end }}

// After (Azure-only):
{{ define `cloud_specific_node_type_id` -}}
    Standard_D3_v2
{{- end }}
```

### Step 5: Simplify `library/functions.tmpl`

Hardcode doc link generation to your cloud:

```go
{{ define "generate_doc_link" -}}
    https://learn.microsoft.com/azure/databricks/{{ .path }}
{{- end }}
```

### Step 6: Update `template/update_layout.tmpl`

Always skip directories/files for unused code paths:

```go
// Skip CI/CD dirs you don't use
{{ skip (printf "%s/%s" $root_dir ".gitlab") }}

// Skip ML code paths you don't use
{{ skip (printf "%s/%s/%s" $root_dir $proj "training/notebooks/Train.py") }}
```

### Step 7: Test

```bash
# Quick preview
databricks bundle init . --output-dir /tmp/test-output

# Run existing tests (if you kept them)
pip install -r dev-requirements.txt
pytest tests -vv
```

### Tips

- **Always keep the upstream test UUID** (`27896cf3-bb3e-476e-8129-96df0406d5c7`) logic in `update_layout.tmpl` if you want tests to work
- **Go template whitespace** matters — use `{{-` and `-}}` to trim surrounding whitespace
- **The `url` function** parses URLs and lets you extract `.Scheme`, `.Host`, `.Path`
- **The `regexp` function** creates a regex you can call `.ReplaceAllString` on
- **The `skip` function** only works in `update_layout.tmpl` — it removes rendered files after generation
- **`min_databricks_cli_version`** in the schema ensures users have a compatible CLI version
