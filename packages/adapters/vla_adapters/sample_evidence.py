"""Validate positive, task-bound evidence from decoded composition samples.

This validates the evidence envelope at Agent and control-plane boundaries. The
native sample probe remains responsible for decoding and executing transforms.
"""


COMPOSABLE_PROCESSOR = "ComposableDataConfig"


def validate_sample_evidence(plan):
    """Reject a successful check without matching composed-dataset samples.

    Legacy processors do not make a sampled-processing claim and may omit evidence.
    Composition is detected from both resolved datasets and requested bindings, so
    dropping one resolved marker cannot turn a composed dataset into a legacy one.
    """
    datasets = plan.get("datasets") if isinstance(plan, dict) else None
    if not isinstance(datasets, list):
        raise ValueError("missing datasets for decoded sample evidence")
    bindings = plan.get("dataset_bindings")
    for position, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError("invalid dataset for decoded sample evidence")
        processor = dataset.get("processor")
        binding = bindings[position] if isinstance(bindings, list) and position < len(bindings) else None
        requested = binding.get("processor") if isinstance(binding, dict) else None
        composed = ("composition" in dataset or dataset.get("factory") == COMPOSABLE_PROCESSOR
                    or isinstance(processor, dict) and processor.get("name") == COMPOSABLE_PROCESSOR
                    or isinstance(requested, dict) and requested.get("name") == COMPOSABLE_PROCESSOR)
        if not composed:
            continue
        composition, probe = dataset.get("composition"), dataset.get("sample_probe")
        if (not isinstance(composition, dict) or not isinstance(probe, dict)
                or type(probe.get("schema_version")) is not int or probe["schema_version"] != 1
                or probe.get("status") != "verified"
                or not isinstance(dataset.get("path"), str) or not dataset["path"]
                or probe.get("path") != dataset["path"]):
            raise ValueError("missing or mismatched composed dataset sample evidence")
        count, indices, samples = probe.get("checked_samples"), probe.get("sample_indices"), probe.get("samples")
        frames = dataset.get("frames")
        if (type(count) is not int or not 1 <= count <= 3
                or not isinstance(indices, list) or len(indices) != count
                or any(type(index) is not int or index < 0 for index in indices)
                or len(set(indices)) != count
                or type(frames) is not int or frames <= 0 or any(index >= frames for index in indices)
                or not isinstance(samples, list) or len(samples) != count
                or any(not isinstance(sample, dict) or type(sample.get("index")) is not int
                       or sample["index"] != index for sample, index in zip(samples, indices))):
            raise ValueError("incomplete or invalid decoded sample indices")
        horizon, sequence = composition.get("action_horizon"), composition.get("action_sequence_keys")
        if (type(horizon) is not int or horizon <= 0
                or type(probe.get("action_horizon")) is not int or probe["action_horizon"] != horizon
                or not isinstance(sequence, list) or any(not isinstance(key, str) or not key for key in sequence)
                or not isinstance(probe.get("action_sequence_keys"), list)
                or probe["action_sequence_keys"] != sequence):
            raise ValueError("decoded sample action sampling differs from composition")
