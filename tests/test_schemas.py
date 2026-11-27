import pytest
from pydantic import ValidationError

from validator.schemas import (AnalysisResult, Finding, Requirement,
                               RequirementCheck, SetFinding)


def _finding(**kw):
    base = dict(req_id="R-01", criterion="проверяемость", severity="major",
                quote="q", problem="p", fix="f")
    return Finding(**{**base, **kw})


def test_valid_models():
    f = _finding()
    check = RequirementCheck(req_id="R-01", verdict="violations", findings=[f])
    sf = SetFinding(kind="пробел", severity="minor", related_req_ids=["R-01"], quote="q", problem="p", fix="f")
    req = Requirement(report_id="R-01", doc_id=None, section="s", quote="q", short="x")
    result = AnalysisResult(requirements=[req], checks=[check], set_findings=[sf])
    assert result.requirements[0].doc_id is None


@pytest.mark.parametrize("bad", ["high", "Критично", "", "CRITICAL"])
def test_bad_severity_rejected(bad):
    with pytest.raises(ValidationError):
        _finding(severity=bad)
    with pytest.raises(ValidationError):
        SetFinding(kind="k", severity=bad, related_req_ids=[], quote="q", problem="p", fix="f")


def test_bad_verdict_rejected():
    with pytest.raises(ValidationError):
        RequirementCheck(req_id="R-01", verdict="maybe", findings=[])


def test_missing_field_rejected():
    with pytest.raises(ValidationError):
        Requirement(report_id="R-01", section="s", quote="q", short="x")  # нет doc_id
