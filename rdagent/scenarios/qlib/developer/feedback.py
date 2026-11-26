import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent

IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]


def process_results(current_result, sota_result):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # Set the metric as the index
    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # Rename the value column to reflect the result type
    current_df.rename(columns={"0": "Current Result"}, inplace=True)
    sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Filter the combined DataFrame to retain only the important metrics
    filtered_combined_df = combined_df.loc[IMPORTANT_METRICS]

    def format_filtered_combined_df(filtered_combined_df: pd.DataFrame) -> str:
        results = []
        for metric, row in filtered_combined_df.iterrows():
            current = row["Current Result"]
            sota = row["SOTA Result"]
            results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
        return "; ".join(results)

    return format_filtered_combined_df(filtered_combined_df)


def opening_gap_review(exp):
    if os.environ.get("RD_OPENING_GAP") != "1":
        return ""
    guidance = (
        "\n高开减仓评审：在 Reasoning 中分别评价原假设和仓位筛选。报告高开天数、信号和成交差异、"
        "相对逢高开减仓的扣费收益差。按原假设指定的分组检查样本数与09:35至收盘的平均收益。"
        "下表按零分组，仅适用于原假设以零为界时。缺少相应分组或消融实验时写明未验证。"
        "不得仅凭组合IC或收益推断单因子失效、共线或因果关系。\n"
    )
    folder = exp.experiment_workspace.workspace_path
    ledger_path = folder / "auction_daily.csv"
    if not ledger_path.exists():
        return guidance + "本次工作区缺逐日账本，无法核对逐日信号与成交。"
    ledger = pd.read_csv(ledger_path, index_col="date", parse_dates=["date"])
    high = ledger.loc[ledger.gap > 0]
    evidence = {
        "高开天数": len(high),
        "模型发出卖出信号天数": int((high.score < 0).sum()),
        "与逢高开减仓信号不同的日期": high.index[high.score >= 0].strftime("%Y-%m-%d").tolist(),
        "模型实际减仓天数": int(ledger.model_order.ne(0).sum()),
        "与逢高开减仓成交数量不同的日期": ledger.index[
            ledger.model_order != ledger.all_high_order
        ].strftime("%Y-%m-%d").tolist(),
    }
    if exp.result is not None and "gain_over_all_high_pp" in exp.result:
        evidence["相对逢高开减仓扣费收益差（百分点）"] = float(exp.result["gain_over_all_high_pp"])
    factors_path = folder / "combined_factors_df.parquet"
    if factors_path.exists():
        factors = pd.read_parquet(factors_path)["feature"].xs("SH588170", level="instrument").reindex(high.index)
        groups = {}
        for name, values in factors.items():
            summary = {"缺值天数": int(values.isna().sum())}
            for label, mask in [("大于零", values > 0), ("小于零", values < 0), ("等于零", values == 0)]:
                returns = high.loc[mask, "label"].dropna()
                summary[label] = {"有效天数": len(returns), "平均收益(%)": 100 * returns.mean() if len(returns) else None}
            groups[str(name)] = summary
        evidence["高开日因子分组"] = groups
    return guidance + json.dumps(evidence, ensure_ascii=False, allow_nan=False)


class QlibFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )
        usr_prompt += opening_gap_review(exp)

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class QlibModelExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibModelExperiment): The experiment to generate feedback for.
            hypothesis (QlibModelHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            HypothesisFeedback: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            sota_result=SOTA_experiment.result.loc[IMPORTANT_METRICS] if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=exp.result.loc[IMPORTANT_METRICS] if exp.result is not None else "execution failed",
        )
        user_prompt += opening_gap_review(exp)

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response)

        # Call the APIBackend to generate the response for hypothesis feedback
        response_hypothesis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response_hypothesis)
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )
