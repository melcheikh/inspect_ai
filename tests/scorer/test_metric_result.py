import math

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log._log import EvalScore
from inspect_ai.scorer import (
    Metric,
    MetricContext,
    MetricResult,
    SampleScore,
    Score,
    Target,
    get_metric_context,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState


@metric
def subpopulation_fpr() -> Metric:
    """Simulates a metric that internally filters to benign samples only."""

    def metric_fn(scores: list[SampleScore]) -> MetricResult:
        benign_scores = [
            s
            for s in scores
            if s.sample_metadata and s.sample_metadata.get("type") == "benign"
        ]
        if not benign_scores:
            return MetricResult.undefined(
                reason="empty_subpopulation", n=0, of=len(scores)
            )

        false_positives = sum(
            1 for s in benign_scores if s.score.value in (1.0, "C", True)
        )
        fpr = false_positives / len(benign_scores)
        return MetricResult(
            value=fpr,
            n=len(benign_scores),
            of=len(scores),
            metadata={"subpopulation": "benign"},
        )

    return metric_fn


@metric
def planned_accuracy() -> Metric:
    """Calculates accuracy dividing by total planned samples from MetricContext."""

    def metric_fn(
        scores: list[SampleScore], context: MetricContext | None = None
    ) -> MetricResult:
        ctx = context or get_metric_context()
        total_planned = ctx.total_samples if ctx and ctx.total_samples else len(scores)
        correct = sum(1 for s in scores if s.score.value in (1.0, "C", True))
        return MetricResult(
            value=correct / total_planned if total_planned > 0 else float("nan"),
            n=correct,
            of=total_planned,
        )

    return metric_fn


@scorer(metrics=[subpopulation_fpr()])
def dummy_detector():
    async def score(state: TaskState, target: Target) -> Score:
        val = 1.0 if "trigger" in state.output.completion else 0.0
        return Score(value=val, answer=state.output.completion)

    return score


def test_metric_result_scalar_populates_eval_metric() -> None:
    task = Task(
        dataset=[
            Sample(
                input="Sample 1",
                target="ok",
                metadata={"type": "benign"},
            ),
            Sample(
                input="Sample 2",
                target="ok",
                metadata={"type": "benign"},
            ),
            Sample(
                input="Sample 3 (trigger)",
                target="alert",
                metadata={"type": "malicious"},
            ),
        ],
        scorer=dummy_detector(),
    )

    log = eval(task, model="mockllm/model", display="none")[0]
    assert log.results is not None
    assert log.results.scores is not None
    assert len(log.results.scores) == 1

    eval_score = log.results.scores[0]
    metric_entry = eval_score.metrics["subpopulation_fpr"]
    assert metric_entry.value == 0.0
    assert metric_entry.n == 2
    assert metric_entry.of == 3
    assert metric_entry.reason is None
    assert metric_entry.metadata == {"subpopulation": "benign"}


def test_metric_result_undefined_reason_when_subpopulation_empty() -> None:
    # All samples are malicious: benign subpopulation is empty
    task = Task(
        dataset=[
            Sample(
                input="Sample 1 (trigger)",
                target="alert",
                metadata={"type": "malicious"},
            ),
            Sample(
                input="Sample 2 (trigger)",
                target="alert",
                metadata={"type": "malicious"},
            ),
        ],
        scorer=dummy_detector(),
    )

    log = eval(task, model="mockllm/model", display="none")[0]
    assert log.results is not None
    assert log.results.scores is not None

    eval_score = log.results.scores[0]
    metric_entry = eval_score.metrics["subpopulation_fpr"]
    assert math.isnan(metric_entry.value)
    assert metric_entry.n == 0
    assert metric_entry.of == 2
    assert metric_entry.reason == "empty_subpopulation"


def test_metric_context_injection_in_planned_accuracy() -> None:
    @scorer(metrics=[planned_accuracy()])
    def dummy_task_scorer():
        async def score(state: TaskState, target: Target) -> Score:
            return Score(value=1.0)

        return score

    # Dataset with 4 samples, evaluated with planned_accuracy
    task = Task(
        dataset=[Sample(input=f"Question {i}", target="ans") for i in range(4)],
        scorer=dummy_task_scorer(),
    )

    log = eval(task, model="mockllm/model", display="none")[0]
    assert log.results is not None
    assert log.results.scores is not None

    eval_score = log.results.scores[0]
    metric_entry = eval_score.metrics["planned_accuracy"]
    assert metric_entry.value == 1.0
    assert metric_entry.n == 4
    assert metric_entry.of == 4


def test_metric_result_terminal_display_rendering() -> None:
    from inspect_ai._display.core.results import task_scores
    from inspect_ai.log import EvalMetric

    scores = [
        EvalScore(
            name="test_scorer",
            scorer="test_scorer",
            metrics={
                "with_denom": EvalMetric(name="with_denom", value=0.75, n=15, of=20),
                "undefined_with_reason": EvalMetric(
                    name="undefined_with_reason",
                    value=float("nan"),
                    n=0,
                    of=10,
                    reason="empty_subpopulation",
                ),
                "legacy_metric": EvalMetric(name="legacy_metric", value=1.0),
            },
        )
    ]

    tables = task_scores(scores)
    assert len(tables) > 0
