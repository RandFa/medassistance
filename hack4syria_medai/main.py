#!/usr/bin/env python
import warnings
from dotenv import load_dotenv

from datetime import datetime

from hack4syria_medai.crew import Hack4SyriaMedai

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run(input: dict = None):
    """
    Run the crew.
    """
    # try:
    Hack4SyriaMedai().crew().kickoff(inputs=input)
    # except Exception as e:
    #     raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    # load_dotenv(override=True)
    # Process a medical case
    input = {
        "topic": "  امرأة تبلغ من العمر 52 عامًا تعاني من خفقان متقطع ودوار خلال الأسبوعين الماضيين. لديها تاريخ طبي من ارتفاع ضغط الدم ومرض السكري من النوع الثاني. https://geekymedics.com/wp-content/uploads/2024/03/ECG-case-study-5.png",
    }  # "Patient complains of chest pain"
    run(input=input)
