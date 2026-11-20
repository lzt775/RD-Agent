"""Question-driven ETF research using RD-Agent proposals and a host-supplied simulator."""
import json

from rdagent.components.proposal import FactorHypothesisGen
from rdagent.core.proposal import Trace
from rdagent.oai.llm_utils import APIBackend


class ETFProposal(FactorHypothesisGen):
    def __init__(self, scenario, limit):
        super().__init__(scenario)
        self.limit = limit

    def prepare_context(self, trace):
        return {
            "hypothesis_and_feedback": "",
            "RAG": None,
            "hypothesis_output_format": 'JSON object {"candidates":[{"hypothesis":"...",'
                '"rationale":"...", "expression":"..."}]}. Write prose in Chinese.',
            "hypothesis_specification": (
                f"Invent 1 to {self.limit} distinct falsifiable candidates. Every expression is a "
                "boolean condition for REDUCING the discretionary sleeve. Follow the supplied grammar. "
                "Develop your own formulas and thresholds before seeing any returns."
            ),
        }, True

    def convert_response(self, response):
        rows = json.loads(response)["candidates"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= self.limit:
            raise ValueError("Invalid number of proposed candidates")
        plan = []
        for i, row in enumerate(rows, 1):
            for key in ("hypothesis", "rationale", "expression"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Candidate {i} is missing {key}")
            plan.append({**{key: row[key].strip() for key in ("hypothesis", "rationale", "expression")},
                         "id": f"F{i}"})
        return plan


class ETFScenario:
    def __init__(self, description):
        self.description = description

    def get_scenario_all_desc(self, **kwargs):
        return self.description


def propose(question, protocol, limit):
    scenario = ETFScenario(protocol)
    return ETFProposal(scenario, limit).gen(Trace(scenario), {"user_instruction": question})


def review(question, candidates, development, prior=None):
    """Only the development measurements are available to the research agent."""
    prompt = ("Review the preregistered ETF hypotheses using the measured development data. "
              "net_pct and gain_pp are after trading costs. Compare costs, worst month and concentration. "
              "Speak qualitatively in the assessment; the host prints all measured numbers. turnover is "
              "a multiple, not percent. worst_month_pct is a calendar-month return, not drawdown. "
              "Every rolling window includes today and shorter and longer windows can overlap. "
              "Distinguish formula wording from what it measures. Abstain if none improves development "
              "net return or positive results are too concentrated. Avoid causal claims. Select one candidate for historical "
              "validation or abstain using null. Return a JSON object with selected, assessment "
              "(Chinese string), next_question (Chinese string) and continue_research (boolean). "
              "Request a further development round only if a distinct falsifiable idea remains. Do not alter any formula.")
    reply = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=json.dumps(dict(question=question, candidates=candidates, development=development,
                                    previous_feedback=prior), ensure_ascii=False),
        system_prompt=prompt, json_mode=True)
    result = json.loads(reply)
    eligible = {r["id"] for r in development if r["id"] != "baseline" and "error" not in r}
    if "selected" not in result or result["selected"] is not None and result["selected"] not in eligible:
        raise ValueError("Review selected an unknown or failed candidate")
    if any(not isinstance(result.get(key), str) or not result[key].strip() for key in ("assessment", "next_question")):
        raise ValueError("Incomplete experiment review")
    if type(result.get("continue_research")) is not bool:
        raise ValueError("Missing continuation decision")
    return {key: result[key] for key in ("selected", "assessment", "next_question", "continue_research")}
