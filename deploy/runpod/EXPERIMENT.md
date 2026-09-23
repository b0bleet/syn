# Controlled comparison with a reference model

## Status checked 2026-09-22, 02:36 UTC

RunPod returned HTTP 200. Pod `76k5htkobq3etj` was listed as `RUNNING`, using commit
`6c645a5`, Qwen3-8B-Base, anchor off, two epochs and no per-source row cap. This is
the existing anchor-off experiment. No new pod was started or stopped.

The authenticated Hugging Face store contained only the earlier finished pointer runs:

| Run | Transfer-development top-1 | In-domain test top-1 |
|---|---:|---:|
| Qwen3-4B, 20260921-173108 | 0.757 | 0.867 |
| Qwen3-8B, 20260921-172137 | 0.767 | 0.889 |

No result for `20260922-014052` was uploaded at this check. Running status alone does
not establish training progress. Local copies of the finished result files are under
`runs/training-audit/pointers/`; the sanitized status is `runs/training-audit/status.json`.

## Data audit

All five local input files match the store byte-for-byte at revision
`7ebc4fa0f835663a624dbf7c535750e7356c0709`. The training file contains 14,576 rows across
79 source tags. The transfer-development file contains 764 rows. Input hashes and row
counts are saved in `runs/training-audit/data-audit.json`.

The former extra-case generator made 2,576 uniform-target copies and 834 date-count copies.
In **244** of the uniform-target candidates, the removed numeric sentence was a routing
reference explicitly declared irrelevant by the policy. This is a conservative count of
one demonstrated failure mode, not a count of all bad labels.

For example, a policy approves when shipment value is 8–18 OR request value is 5–15,
unless case value is 54. The case says routing reference 901, case value 64, shipment
value 19, and request value 6. Removing routing reference 901 leaves the correct answer
as approval; assigning 50% to rejection introduces a false training target.

There is a second issue even when relevant evidence is removed: a spend policy permits
automatic approval or director sign-off, while an option also says outright rejection.
An unknown amount does not make that third option equally plausible.

Automatic uniform-label generation has been removed. Date copies preserve the original
context and label and remain an opt-in experiment; there are now zero generated uniform
copies. All extra copies are off for the next baseline.

## Next experiment

Obtain the anchor-off checkpoint and its result first. Compare its saved configuration
and data with the next run before attributing a difference to ordinal loss.

The next run changes ordinal-loss weight from 1 to 0. Keep anchor off; keep balancing,
source holdout and date copies off. Preserve the base, input rows, seed, rank, learning
rate, epochs, batch size, augmentation and token limit. The precise preview command is in
[README.md](README.md#next-controlled-pointer-experiment). Commit and push the fixes
before launching: the pod clones a Git commit, not the local working tree.

Balancing, source holdout and date copies can each be tested separately after this result.
Equalizing 79 source tags is not the same as equalizing broad task families: 60 tags here
are generated composition types. Holding out a single small source is also a hypothesis,
not a proven better model-selection rule.

## Evaluation

Use `syn bench` to send identical converted examples to the SYN and reference endpoints.
It now reports per-source metrics, paired bootstrap intervals, failed rows and dataset
hashes, and rejects mismatched paired examples. Require full coverage before a win claim.
Record the model/checkpoint revisions; use a fresh benchmark output directory per pair.

This is a shared serving protocol, not a reproduction of the reference model's published native-suite
numbers: binary questions are represented as choice and each request contains one question.
Do not substitute published aggregate scores for paired predictions. Row bootstrap
intervals also do not measure variation across training seeds or entirely new sources.

The repeatedly inspected transfer-development set is suitable for development comparisons.
Choose all settings before evaluating an untouched transfer test. No new model benchmark
or accuracy improvement has been measured as part of this code change.
