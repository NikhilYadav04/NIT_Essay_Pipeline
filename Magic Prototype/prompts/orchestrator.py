from prompts.base import BasePrompts
from prompts.agents import GREAgentPrompts

class GREOrchestratorPrompts(BasePrompts):
    system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on other expert grader's scores and reasoning.
Please provide a numerical score for the provided essay according to the opinions of the other expert grader's scores and reasoning.
Each expert grader is an expert grader for a specific aspect of the essay.

- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.
- You will carefully read each expert grader's score and reasoning, prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score balancing the opinions of the other expert grader's scores and reasoning.
- You will provide a detailed explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve their essay, balancing the opinions of the other expert grader's feedback.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

The expert grader's scores and reasoning are as follows:
{expert_grader_scores_and_reasoning}

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given expert grader's scores and reasoning, prompt and student essay carefully and score the <student_essay>.
Provide an integer score between 0 and 6 by balancing the provided expert grader's scores and reasoning.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}

"""

    

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "system_prompt": cls.system_prompt,
            "input_prompt": cls.input_prompt,
        }

    @classmethod
    def format_prompt_inference(cls, grading_instruction: dict, domain_scores: list[int], domain_feedbacks: list[str]) -> str:
        expert_grader_scores_and_reasoning = ""
        for domain_score, domain_feedback, aspect_rubric in zip(domain_scores, domain_feedbacks, GREAgentPrompts.aspect_rubrics):
            aspect_name = aspect_rubric[2]
            expert_grader_scores_and_reasoning += f"""
<expert_grader_judgement>
{aspect_name}
{{
    "score": {domain_score},
    "examiner_comment": {domain_feedback}
}}
</expert_grader_judgement>\n
"""

        system_prompt_formatted = cls.system_prompt.format(
            expert_grader_scores_and_reasoning=expert_grader_scores_and_reasoning,
            prompt=grading_instruction["prompt"],
            # task_directions=grading_instruction["task_directions"],
            output_format=cls.output_format,
            min_score=0,
            max_score=6
        )
        input_prompt_formatted = cls.input_prompt.format(
            essay_text=grading_instruction["essay_text"]
        )

        return cls.alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")