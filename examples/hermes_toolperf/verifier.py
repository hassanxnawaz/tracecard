"""Success checks for the Hermes ToolPerf battery (Nous Research).

Ported from SUCCESS in hermes-toolperf-evals/abeval/ab_eval.py. Two upstream
checks also read files the agent wrote in its sandbox (err_replay_patch,
err_multi_dir). Those sandboxes are not in the published data, so here they
are checked from the agent's final output only. That is the one known
difference from the upstream scorer.
"""

CHECKS = {
    "err_python_env": lambda t: "ENV_OK_4477" in t,
    "err_replay_patch": lambda t: "CONFIG_OK_881" in t,
    "err_ambiguous_edit": lambda t: "HANDLERS_OK_552" in t,
    "err_case_search": lambda t: "settings.ini" in t and "client.go" in t,
    "err_hidden_search": lambda t: "rotation.cfg" in t and "ops.md" in t,
    "err_big_output": lambda t: "tok_9f31c_middle" in t,
    "err_multi_dir": lambda t: "1.4.2,0.9.7,3.2.1" in t,
    "err_inline_script": lambda t: "21341334000" in t.replace(",", ""),
    "err_big_file_read": lambda t: "X99Q" in t,
}


def verify(run, meta):
    check = CHECKS.get(run.task)
    if check is None:
        return None
    return bool(check(meta.get("tail", "") or run.output or ""))
