import json
import os
import re
import shlex
from uuid import uuid4
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QTDockerEnv


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)
        provider = os.environ.get("RD_QLIB_PROVIDER")
        if provider:
            for name, code in list(self.file_dict.items()):
                if name.endswith(".yaml"):
                    configured = code.replace('"~/.qlib/qlib_data/cn_data"', json.dumps(provider))
                    if os.name == "nt":
                        configured = configured.replace("qlib_init:\n", "qlib_init:\n    kernels: 1\n", 1)
                    if symbols := os.getenv("RD_QLIB_SYMBOLS"):
                        basket = json.loads(symbols)
                        configured = configured.replace("market: &market csi300", "market: &market " + json.dumps(basket))
                        k = min(50, max(1, len(basket) // 2))
                        configured = configured.replace("topk: 50", f"topk: {k}").replace("n_drop: 5", f"n_drop: {min(5,k)}")
                        if len(basket) == 1:
                            configured = configured.replace("        - class: CSZScoreNorm\n          kwargs:\n              fields_group: label", "")
                    elif market := os.getenv("RD_QLIB_MARKET"):
                        configured = configured.replace("market: &market csi300", "market: &market " + json.dumps(market))
                    if os.getenv("RD_ETF_PROTOCOL"):
                        # Cross-sectional label normalization makes a single ETF label undefined.
                        configured = configured.replace("        - class: CSZScoreNorm\n          kwargs:\n              fields_group: label", "")
                        configured = configured.replace("Ref($close, -2)/Ref($close, -1)", "Ref($open, -2)/Ref($open, -1)")
                        configured = configured.replace("n_jobs: 20", "n_jobs: 0")
                        # The stock-universe penalties suppress all splits on a few hundred raw ETF returns.
                        configured = configured.replace("lambda_l1: 205.6999", "lambda_l1: 0.0").replace(
                            "lambda_l2: 580.9768", "lambda_l2: 1.0")
                        configured = configured.split("        - class: SigAnaRecord", 1)[0]
                        configured += '\nexperiment_name: "{{ etf_exp_name }}"\n'
                    self.inject_files(**{name: configured})

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs) -> str:
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv()
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            logger.error(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
            return None, "Unknown environment type"
        qtde.prepare()

        if os.getenv("RD_ETF_PROTOCOL"):
            # Each prediction belongs to one recorder. Stop immediately on failure so old metrics cannot be reused.
            run_env = {"PYTHONPATH": "./", **run_env, "etf_exp_name": "etf_" + uuid4().hex}
            if "batch_size" in run_env:
                # Qlib drops the final incomplete batch in both training and validation.
                run_env["batch_size"] = str(min(16, int(run_env["batch_size"])))
            worker = shlex.quote(Path(os.environ["RD_ETF_ACCOUNT_WORKER"]).as_posix())
            result = qtde.run(local_path=str(self.workspace_path), env=run_env,
                              entry=f"qrun {shlex.quote(qlib_config_name)} && python {worker}")
            path = self.workspace_path / "qlib_res.csv"
            if result.exit_code or not path.is_file():
                logger.error(result.stdout[-3000:])
                return None, result.stdout
            return pd.read_csv(path, index_col=0).iloc[:, 0], result.stdout

        # Run the Qlib backtest
        execute_qlib_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env,
        )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        execute_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env,
        )

        quantitative_backtesting_chart_path = self.workspace_path / "ret.parquet"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_parquet(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            logger.error("No result file found.")
            return None, execute_qlib_log + "\n" + execute_log

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            # Here, we ensure that the qlib experiment has run successfully before extracting information from execute_qlib_log using regex; otherwise, we keep the original experiment stdout.
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            execute_qlib_log = "\n".join(matches)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            return None, execute_qlib_log
