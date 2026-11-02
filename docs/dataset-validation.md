# Dataset registration and reusable validation

Registering a DATASET now requests background source validation when its machine
has a compatible Agent and one active native OpenPI profile. Multiple profiles
require an explicit selection in asset details. Offline or older Agents retain
the registration and show the reason validation cannot yet run.

The source check reads metadata and file manifests, verifies all timestamp and
episode rows under the registered FPS, and decodes the first, middle and last
raw records. It also prepares the local LeRobot/Hugging Face cache. It does not
choose a training composition, action horizon, normalization layout or model.
Validation waits while the machine has an active training attempt.

Asset details expose the latest report and a revalidation action through:

- `POST /api/v1/assets/{asset_id}/validations` with an optional `profile_id`.
- `GET /api/v1/assets/{asset_id}/validation`.

Source status is independent of the asset's registration/availability state.
Reports use `PENDING`, `SUCCEEDED`, `FAILED`, `STALE` or `BLOCKED`. PENDING means
queued or executing; the protocol does not claim a separate start acknowledgement.
The server owns the internal report namespace and rejects browser attempts to
submit it. Old acknowledgements cannot replace reports for changed registrations.

## Reuse during training

The first training preview still verifies the selected composition and model
input contract. Successful timestamp, raw-source and composed-sample evidence is
saved separately under the Agent's private `state/dataset-validation-cache`.
Runner startup and the training entrypoint continue to run their normal checks,
but reuse evidence when the source and processing fingerprints still match.

Fingerprints include canonical data paths, metadata content, source-file versions
(including device, inode, size, mtime and ctime), relevant Python source, runtime
library versions and effective FPS. Composed samples also bind the normalization
file, selected processor, composition and model input settings. Changing data,
normalization, processing code, environment or the relevant configuration causes
a fresh check. Corrupt or unsafe cache files fall back to full validation.

Cache hits return identical evidence; cache timing and hit markers are kept out
of the execution plan so the user's confirmed configuration digest stays stable.
Model/checkpoint bindings, source changes, GPU capacity, writable paths and output
space are still checked on every launch. These checks authorize data compatibility,
not dataset semantic quality or model convergence.

## Rollout

The feature needs the updated control plane, web bundle and an Agent advertising
`DATASET_VALIDATE`. An active machine running an older Agent shows a blocked
source-check status until its Agent can be upgraded while idle. Do not replace
the Agent or adapter environment underneath an active training process.
