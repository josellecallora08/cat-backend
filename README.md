# CAT Backend

Backend services for the Collection Agent Trainer platform.

## Dynamic rubric seed flow

The dynamic rubric seed flow synchronizes approved rubric definitions from a strict, local JSON source into the campaign-owned negotiation-standard tables. It is implemented by `scripts/seed_rubrics.py` and is disabled by default.

### Source format

Only a bounded local file with a `.json` extension is supported. Remote URLs, connection strings, and other content types are rejected. The document must be a JSON object with no unexpected top-level fields:

```json
{
  "source_id": "approved-rubrics-placeholder-v1",
  "definitions": [
    {
      "campaign_id": "00000000-0000-0000-0000-000000000000",
      "name": "Collections Quality Rubric",
      "description": "Placeholder rubric description.",
      "draft_content": {
        "schema_version": 1,
        "overall_passing_score": 70,
        "blocks": []
      },
      "publish": false,
      "publication_note": "Placeholder publication note."
    }
  ]
}
```

`source_id` and `definitions` are required. Each definition requires a valid `campaign_id`, a non-empty `name`, and `draft_content` matching the strict `NegotiationStandardContent` schema. `description`, `publish`, and `publication_note` are optional. Unknown fields, invalid UUIDs, malformed UTF-8/JSON, invalid rubric content, and missing required fields are rejected without persisting that definition. Empty `definitions` is valid and produces a successful run with zero record changes.

Names are trimmed and internal whitespace is normalized for display. Matching uses the case-folded normalized name together with `source_id`, forming the stable identity `(source_id, normalized rubric name)`. Do not use credentials, tokens, or private connection details in `source_id`, names, descriptions, or publication notes.

### Fingerprints, idempotency, and versions

Rubric content is validated, normalized, serialized deterministically, and fingerprinted with the existing canonical SHA-256 hashing behavior. A matching identity reuses its existing `NegotiationStandard`; a missing identity creates one. A matching `(standard, content_hash)` reuses its immutable `NegotiationStandardVersion`. Changed content creates the next immutable version number and preserves every earlier snapshot.

Running the same source repeatedly is idempotent: after the first successful run, later runs create no additional standards or versions. Database uniqueness constraints protect seeded identities and content fingerprints during concurrent runs. Each definition is processed in its own transaction, so valid definitions can commit even when another definition is rejected; persistence failures roll back that definition and fail the run.

A definition with `publish: true` must pass publication validation before the current pointer changes. The standard has at most one current published version. Publishing replaces the pointer atomically; it does not modify or delete the prior version. Publishing the same content again is a no-op after the existing version is reused.

### Configuration

Copy the placeholders in `.env.example` and set values through the approved environment or secret configuration. The supported settings are:

| Setting | Default | Description |
| --- | --- | --- |
| `CAT_RUBRIC_SEED_ENABLED` | `false` | Enables seeding. Keep disabled unless an approved source is available. |
| `CAT_RUBRIC_SOURCE` | empty | Required when enabled; path to the bounded local JSON file. |
| `CAT_RUBRIC_SOURCE_ID` | empty | Optional safe override for the document's `source_id`. |
| `CAT_RUBRIC_SOURCE_MAX_BYTES` | `1048576` | Maximum source size in bytes; must be positive. |

When `CAT_RUBRIC_SEED_ENABLED=false`, seeding is skipped, no source is read, and no database rows are changed. The result status is `disabled`. When enabled, a missing or unreadable source, invalid size limit, malformed JSON, or unsupported source produces a safe `failed` result. Source paths and secret values are not included in results, logs, or error messages.

### Standalone invocation

From `cat-backend`, run:

```bash
python -m scripts.seed_rubrics
```

The command opens the configured async database session, runs the seed flow, and writes one safe JSON result to stdout. It returns exit code `0` for `success`, `partial_success`, or `disabled`, and a non-zero exit code for `failed`. It never prints source payloads, SQL, credentials, tokens, tracebacks, or internal connection details.

Startup integration is optional. If an existing startup caller invokes `run_seed()`, it must honor `CAT_RUBRIC_SEED_ENABLED` and use the same result contract. No startup behavior is changed while seeding remains disabled.

### Permissions and secrets

Standalone operation is intended for an administrator-controlled environment. An externally exposed adapter must authenticate the caller and enforce the `admin` role before reading the source or opening a mutation transaction; it must not trust a client-supplied role. Source credentials, if a future approved provider requires them, must come from `CAT_` environment variables or the platform secret provider. Never commit real credentials or place them in this document or a source JSON file.

Audit records contain only the run ID, safe source ID, initiating principal when known, timestamp, status, and aggregate counts. They do not contain passwords, JWTs, API keys, source payloads, or filesystem/connection details.

### Result and output contract

Each run returns a `SeedRunResult` with:

- `run_id` and safe `source_id`;
- `status`: `success`, `partial_success`, `failed`, or `disabled`;
- `reused_rubrics`, `created_rubrics`, `reused_versions`, `created_versions`, `published_versions`, and `rejected_definitions`;
- safe `warnings`; and
- definition outcomes containing a non-secret identity, status, optional version ID, publication flag, and classified safe error.

A `partial_success` result means at least one definition succeeded and at least one was rejected. Validation rejections identify the definition by its safe source/name identity and use a field-safe validation message. Source and persistence failures identify only the failed phase and remediation category.

### Troubleshooting

- **`disabled`**: set `CAT_RUBRIC_SEED_ENABLED=true` only when seeding is intended.
- **Missing source/configuration failure**: set `CAT_RUBRIC_SOURCE` to an existing local `.json` file and verify `CAT_RUBRIC_SOURCE_MAX_BYTES` is positive.
- **Malformed or unsupported source**: validate UTF-8 JSON, use an object with exactly `source_id` and `definitions`, and remove unknown fields. Do not use a URL.
- **Definition rejected**: check the named definition's UUID, required fields, strict rubric schema, and publication requirements. Other independent valid definitions may still commit.
- **Synchronization conflict**: verify that one campaign is not being assigned different seeded identities and retry after the competing run finishes.
- **Unexpected persistence failure**: inspect the database operationally without exposing SQL or credentials; the affected definition is rolled back.

### Migration compatibility and pinned versions

The seed migration adds nullable `source_id` and `source_rubric_key` fields so legacy standards remain addressable without guessed source identities. Existing standards are not silently claimed by the seed flow. It also enforces uniqueness for seeded identity pairs and `(standard_id, content_hash)` while retaining version-number uniqueness. Existing session and evaluation foreign keys remain unchanged.

Published and historical versions are immutable snapshots. New seeds replace only the standard's current-version pointer; they do not delete or rewrite versions referenced by existing sessions or evaluations. New sessions resolve and store the current `NegotiationStandardVersion` ID at creation. Evaluations store the exact version ID and snapshot used for scoring, so later publication cannot change historical results.

To verify pinning, record the version ID from the seed result, create a session/evaluation for the campaign, and compare its stored `negotiation_standard_version_id` with that ID. Seed and publish a changed definition, then confirm the original session/evaluation still points to the original version while new sessions resolve the new current version. Query the standard's current version for the one published pointer and query the version table by ID to inspect the immutable snapshot and fingerprint; do not infer a historical result from the current pointer alone.

### Bundled concrete standalone seed

With no `CAT_RUBRIC_SOURCE` configured, `python -m scripts.seed_rubrics` runs the bundled concrete setup. It looks up (but never creates) active `admin@cat.ph` and `agent@cat.ph` users, then creates or reuses `Dynamic Rubric Test Campaign`, the active collections scenario `Temporary Financial Hardship Payment Arrangement`, its agent assignment, and a three-block published rubric weighted Call Opening 25%, Empathy and Communication 35%, and Negotiation and Resolution 40%. It creates or reuses one pending session pinned to the immutable published version. The JSON result additionally includes `campaign_id`, `scenario_id`, `agent_id`, `rubric_version_id`, and `session_id`. No transcript or evaluation data is generated. Ensure the normal user seeder has run first so both active accounts exist; otherwise the command returns a safe configuration failure.
