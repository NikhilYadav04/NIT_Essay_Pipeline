from prompts.base import BasePrompts


class ASAPAgentPrompts(BasePrompts):
    """
    3-agent rubric tuned for ASAP (middle/high school persuasive essays).
    Replaces the 5-agent GRE rubric which over-penalises student writing.

    Agent 1: Content & Task Response   (was GRE agents 1+2)
    Agent 2: Organisation & Development (was GRE agent 3)
    Agent 3: Language & Conventions     (was GRE agents 4+5)
    """

    # ── Rubric texts ───────────────────────────────────────────────────────────

    aspect_1_rubric = """
Aspect 1: Content & Task Response
Score 6: Clearly addresses the prompt with a well-developed position; strong, relevant supporting details.
Score 5: Addresses the prompt with a clear position; adequate supporting details.
Score 4: Addresses the prompt; position is present but development is limited or uneven.
Score 3: Partially addresses the prompt; position is vague or support is weak/irrelevant.
Score 2: Minimally addresses the prompt; position is unclear or unsupported.
Score 1: Little or no evidence of understanding the prompt or task.
Score 0: Off-topic, blank, copied text, or incomprehensible.
"""

    aspect_2_rubric = """
Aspect 2: Organisation & Development
Score 6: Clear, logical structure with introduction, body, and conclusion; ideas connect smoothly.
Score 5: Organised response with mostly logical flow; transitions are present.
Score 4: Adequate organisation; some lapses in flow or transitions.
Score 3: Limited organisation; ideas are loosely connected or repetitive.
Score 2: Poor organisation; hard to follow; little sense of structure.
Score 1: No discernible organisation; response is extremely brief or incoherent.
Score 0: Off-topic, blank, copied text, or incomprehensible.
"""

    aspect_3_rubric = """
Aspect 3: Language & Conventions
Score 6: Fluent and varied sentences; precise word choice; few or no errors in grammar/spelling/punctuation.
Score 5: Clear language; some variety in sentence structure; minor errors that do not affect meaning.
Score 4: Acceptable language control; errors are present but do not frequently obscure meaning.
Score 3: Noticeable errors in grammar, spelling, or punctuation that sometimes interfere with meaning.
Score 2: Frequent errors that often obscure meaning; limited vocabulary.
Score 1: Pervasive errors making the response difficult to understand.
Score 0: Off-topic, blank, copied text, or incomprehensible.
"""

    aspect_rubrics = [
        ("content",      aspect_1_rubric, "Aspect 1: Content & Task Response"),
        ("organisation", aspect_2_rubric, "Aspect 2: Organisation & Development"),
        ("language",     aspect_3_rubric, "Aspect 3: Language & Conventions"),
    ]

    # ── System prompts ─────────────────────────────────────────────────────────

    content_system_prompt = """
You are an expert essay grader scoring a student persuasive essay based on the rubric below.
You specialise in evaluating Content & Task Response — whether the student answered the prompt
with a clear position and relevant supporting details.

- The length matters: a well-developed essay should have at least 2-3 paragraphs.
- Read the rubric, prompt, and essay carefully.
- Reason step-by-step before giving your score.
- A low score is not harmful — an accurate score helps the student improve.

Rubric:
<content_rubric>
{content_rubric}
</content_rubric>

Prompt:
<essay_prompt>
{prompt}
</essay_prompt>

{output_format}
"""

    organisation_system_prompt = """
You are an expert essay grader scoring a student persuasive essay based on the rubric below.
You specialise in evaluating Organisation & Development — whether the essay has clear structure,
logical flow, and well-connected ideas.

- The length matters: a well-developed essay should have at least 2-3 paragraphs.
- Read the rubric, prompt, and essay carefully.
- Reason step-by-step before giving your score.
- A low score is not harmful — an accurate score helps the student improve.

Rubric:
<organisation_rubric>
{organisation_rubric}
</organisation_rubric>

Prompt:
<essay_prompt>
{prompt}
</essay_prompt>

{output_format}
"""

    language_system_prompt = """
You are an expert essay grader scoring a student persuasive essay based on the rubric below.
You specialise in evaluating Language & Conventions — grammar, spelling, punctuation, vocabulary,
and sentence variety.

- Read the rubric, prompt, and essay carefully.
- Reason step-by-step before giving your score.
- A low score is not harmful — an accurate score helps the student improve.

Rubric:
<language_rubric>
{language_rubric}
</language_rubric>

Prompt:
<essay_prompt>
{prompt}
</essay_prompt>

{output_format}
"""

    # ── Formatting ─────────────────────────────────────────────────────────────

    @classmethod
    def format_prompt_inference(cls, grading_instruction: dict, agent_rubric_type: str, current_aspect_rubric: str) -> str:
        prompt = grading_instruction["prompt"]
        essay_text = grading_instruction["essay_text"]

        if agent_rubric_type == "content":
            system_prompt_formatted = cls.content_system_prompt.format(
                content_rubric=current_aspect_rubric,
                prompt=prompt,
                output_format=cls.output_format,
            )
        elif agent_rubric_type == "organisation":
            system_prompt_formatted = cls.organisation_system_prompt.format(
                organisation_rubric=current_aspect_rubric,
                prompt=prompt,
                output_format=cls.output_format,
            )
        else:
            system_prompt_formatted = cls.language_system_prompt.format(
                language_rubric=current_aspect_rubric,
                prompt=prompt,
                output_format=cls.output_format,
            )

        input_prompt_formatted = cls.input_prompt.format(essay_text=essay_text)
        return cls.alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "content_system_prompt":      cls.content_system_prompt,
            "organisation_system_prompt": cls.organisation_system_prompt,
            "language_system_prompt":     cls.language_system_prompt,
            "input_prompt":               cls.input_prompt,
            "aspect_rubrics":             cls.aspect_rubrics,
        }
