"""Research capabilities for the Quantumn and TradingAgents-CN ETF integrations."""
import json
import os
import subprocess


def backtest(project, python, plan, phase):
    if phase not in ('development', 'validation'):
        raise ValueError('Unknown experiment phase')
    subprocess.run([str(python), str(project / "scripts/etf_explore_backtest.py"), str(plan), phase],
                   cwd=project, check=True)
    return json.loads((plan.parent / phase / "metrics.json").read_text(encoding="utf-8"))


def independent_review(python, fork, credentials, experiment):
    return trading_research(python, fork, credentials, experiment, "review_experiment", "assessment")


def suggest_strategy(python, fork, credentials, experiment):
    return trading_research(python, fork, credentials, experiment, "suggest_strategy", "idea")


def trading_research(python, fork, credentials, experiment, action, field):
    runner = fork / "cli/local_etf_research.py"
    if not runner.is_file() or not python.is_file():
        raise ValueError("请检查 TradingAgents-CN 仓库位置和 tradingagents-cn 解释器")
    env = dict(os.environ, PYTHONUTF8="1")
    for key in ("ETF_LLM_BASE_URL", "ETF_LLM_MODEL", "ETF_LLM_API_KEY", "ETF_LLM_MAX_TOKENS", "ETF_LLM_THINKING"):
        if credentials.get(key):
            env.setdefault(key, credentials[key])
    response = subprocess.run([str(python), str(runner)], cwd=fork, env=env, encoding="utf-8",
                              input=json.dumps({"action": action, "experiment": experiment}),
                              capture_output=True, timeout=420)
    if response.returncode:
        raise ValueError(f"TradingAgents-CN {action} 失败，请检查模型配置、网络和项目版本")
    review = json.loads(response.stdout)
    if not isinstance(review.get(field), str) or not review[field].strip():
        raise ValueError(f"TradingAgents-CN 缺少 {field} 正文")
    return review
