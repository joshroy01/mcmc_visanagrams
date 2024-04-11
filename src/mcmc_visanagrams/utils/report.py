from typing import List
from pathlib import Path
import subprocess
import pprint


def make_report(config, saved_img_paths: List[Path], stage_num: int = 1):
    stage_output_path = config.stage_1_output_path
    report_stage_1_path = stage_output_path / "report.md"
    make_md_report(report_stage_1_path, config, saved_img_paths, stage_num)
    make_pdf_report(report_stage_1_path)


def make_md_report(report_md_path: Path, config, saved_img_paths: List[Path], stage_num: int = 1):
    lines = [f"# Stage {stage_num} Report", ""]

    config_lines = ["## Config", "", "```yaml"]
    config_lines.append(pprint.pformat(config._config_dict))
    config_lines.append("```\n")
    lines.extend(config_lines)

    output_lines = ["## Output", ""]
    stage_1_output_lines = [f"![View 1]({img_path.as_posix()})" for img_path in saved_img_paths]
    output_lines.extend(stage_1_output_lines)

    lines.extend(output_lines)

    with report_md_path.open("w") as f:
        f.write("\n".join(lines))


def make_pdf_report(report_md_path: Path):
    report_pdf_path = report_md_path.with_suffix(".pdf")
    cmd = [
        "conda", "run", "-n", "mcmc_visanagrams", "mdpdf", "-o",
        report_pdf_path.as_posix(),
        report_md_path.as_posix()
    ]
    subprocess.call(cmd)
