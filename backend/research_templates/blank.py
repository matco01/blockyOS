"""Replace this entrypoint with your architecture, trainer or evaluator."""
from blocky_research import Run

run = Run()
run.progress("Project started. Implement your training/evaluation loop here.")
# Configuration is available as run.config, and artifacts belong in run.directory.
# Report only actual measurements, e.g.:
# run.metric('accuracy', measured_score, unit='fraction', protocol='my-evaluator/1',
#            task='my-task', dataset_sha256=actual_dataset_hash,
#            scope='held-out examples; exact match', better='higher', samples=per_example_scores)
print("No evaluator configured yet; no scores were reported.")
