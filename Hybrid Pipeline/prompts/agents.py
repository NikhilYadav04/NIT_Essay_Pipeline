"""
hybrid_prototype/prompts/agents.py
MAGIC agents modified for hybrid architecture.

THE ONE CHANGE vs. Magic Pipeline:
  - format_prompt_inference() accepts a 4th parameter: evidence_json: str = ""
  - When evidence_json is provided, each agent's system prompt receives an
    <evidence_record> block and an instruction to use it as its factual foundation.
  - All rubric definitions and aspect_rubrics are IDENTICAL to Magic Pipeline.
"""
from prompts.base import BasePrompts

class GREAgentPrompts(BasePrompts):
    aspect_1_rubric = """
Aspect 1: Quality of the response to the prompt instructions
Score 6: The essay articulates a clear and insightful position on the issue in accordance with the assigned task.
Score 5: The essay presents a clear and well-considered position on the issue in accordance with the assigned task.
Score 4: The essay presents a clear position on the issue in accordance with the assigned task.
Score 3: The essay is vague or limited in addressing the specific task directions and/or in presenting or developing a position on the issue.
Score 2: The essay is unclear or seriously limited in addressing the specific task directions and/or in presenting or developing a position on the issue.
Score 1: The essay presents little or no understanding of how to respond to the prompt.
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""
    aspect_2_rubric = """
Aspect 2: Considering the complexities of the issue
Score 6: The essay develops the position fully, with compelling reasons and/or persuasive examples.
Score 5: The essay develops the position with logically sound reasons and/or well-chosen examples.
Score 4: The essay develops the position with relevant reasons and/or examples.
Score 3: The essay is weak in the use of relevant reasons or examples, or relies largely on unsupported claims.
Score 2: The essay provides few, if any, relevant reasons or examples in support of its claims.
Score 1: The essay provides little or no evidence of understanding the issue.
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""
    aspect_3_rubric = """
Aspect 3: Organizing, developing, and expressing ideas
Score 6: The essay sustains a well-focused, well-organized analysis, connecting ideas logically.
Score 5: The essay is focused and generally well organized, connecting ideas appropriately.
Score 4: The essay's ideas are adequately focused and organized.
Score 3: The essay is limited in focus and/or organization.
Score 2: The essay is poorly focused and/or poorly organized.
Score 1: The essay provides little or no evidence of the ability to develop an organized response (e.g., is disorganized and/or extremely brief).
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""
    aspect_4_rubric = """
Aspect 4: Vocabulary and sentence variety
Score 6: The essay conveys ideas fluently and precisely, using effective vocabulary and sentence variety.
Score 5: The essay conveys ideas clearly and well, using appropriate vocabulary and sentence variety.
Score 4: The essay conveys ideas with acceptable clarity, demonstrating sufficient control of language.
Score 3: The essay has problems in language and sentence structure that result in a lack of clarity.
Score 2: The essay has serious problems in language and sentence structure that frequently interfere with meaning.
Score 1: The essay has severe problems in language and sentence structure that persistently interfere with meaning.
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""
    aspect_5_rubric = """
Aspect 5: Grammar and mechanics
Score 6: The essay demonstrates superior facility with the conventions of standard written English (i.e., grammar, usage, and mechanics) but may have minor errors.
Score 5: The essay demonstrates facility with the conventions of standard written English but may have minor errors.
Score 4: The essay generally demonstrates control of the conventions of standard written English but may have some errors.
Score 3: The essay contains occasional major errors or frequent minor errors in grammar, usage, or mechanics that can interfere with meaning.
Score 2: The essay contains serious errors in grammar, usage, or mechanics that frequently obscure meaning.
Score 1: The essay contains pervasive errors in grammar, usage, or mechanics that result in incoherence.
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""

    aspect_rubrics = [
        ("argumentative", aspect_1_rubric, "Aspect 1: Quality of the response to the prompt instructions"),
        ("argumentative", aspect_2_rubric, "Aspect 2: Considering the complexities of the issue"),
        ("argumentative", aspect_3_rubric, "Aspect 3: Organizing, developing, and expressing ideas"),
        ("vocabulary",    aspect_4_rubric, "Aspect 4: Vocabulary and sentence variety"),
        ("grammar",       aspect_5_rubric, "Aspect 5: Grammar and mechanics"),
    ]

    # ── HYBRID ADDITION: evidence instruction injected into all three prompts ──
    # When evidence_json is provided, this text is injected into the agent prompt.
    # It anchors the agent to the SRCE evidence record instead of re-reading freely.
    _EVIDENCE_INSTRUCTION = (
        "- You have been provided a structured evidence record (<evidence_record>) extracted "
        "from the student essay by a dedicated analysis agent.\n"
        "- Use this evidence record as your PRIMARY factual foundation when scoring.\n"
        "- The evidence record tells you what evidence exists for each rubric criterion — "
        "you do not need to search for it yourself.\n"
        "- Base your score on the evidence in this record. You may refer to the original essay "
        "only to resolve an inconsistency in the evidence record.\n"
        "- Do NOT re-evaluate the essay from scratch — the evidence has already been extracted."
    )

    _NO_EVIDENCE_INSTRUCTION = (
        "- Read the essay carefully and identify evidence for each rubric criterion yourself."
    )

    # ── System prompts: use string replacement (not .format) for rubric/prompt/output
    #    so that the {evidence_instruction} and {evidence_record} placeholders can be
    #    injected first via .format(), before the per-rubric replacements are done.
    # ──────────────────────────────────────────────────────────────────────────────

    argumentative_system_prompt = (
        "You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric.\n"
        "You specialize in scoring the argumentative qualities of an essay.\n"
        "Please provide a numerical score for the provided essay considering all aspects of the specified rubric.\n\n"
        "- Provide an appropriate holistic argumentative score.\n"
        "- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.\n"
        "- You will carefully read the rubric (<argumentative_rubric>), prompt (<essay_prompt>), "
        "student essay (<student_essay>), and evidence record (<evidence_record>).\n"
        "{evidence_instruction}\n"
        "- You will reason carefully as to why you chose this score following the rubric and guidelines.\n"
        "- You will provide a detailed step-by-step explanation of your reasoning for the score.\n"
        "- You will provide feedback for the student on how to improve the argumentative qualities of their essay.\n"
        "- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student "
        "improve their score in future essays.\n\n"
        "The rubric or rubrics for this essay is as follows:\n"
        "<argumentative_rubric>\n"
        "ARGUMENTATIVE_RUBRIC_PLACEHOLDER\n"
        "</argumentative_rubric>\n\n"
        "The structured evidence record extracted from the student essay:\n"
        "<evidence_record>\n"
        "EVIDENCE_RECORD_PLACEHOLDER\n"
        "</evidence_record>\n\n"
        "The prompt is as follows:\n"
        "<essay_prompt>\n"
        "PROMPT_PLACEHOLDER\n"
        "</essay_prompt>\n\n"
        "Review the given rubric, evidence record and prompt carefully and score the <student_essay>.\n"
        "Provide a numerical score by using the provided rubric's guidance. The score should be a number between 0 and 6.\n"
        "Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student "
        "improve their score in future essays.\n\n"
        "OUTPUT_FORMAT_PLACEHOLDER\n"
    )

    vocabulary_system_prompt = (
        "You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric.\n"
        "You specialize in scoring the vocabulary and sentence variety of an essay.\n"
        "Please provide a numerical score for the provided essay considering all aspects of the specified rubric.\n\n"
        "- Provide an appropriate holistic vocabulary score.\n"
        "- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.\n"
        "- You will carefully read the rubric (<vocabulary_rubric>), prompt (<essay_prompt>), "
        "student essay (<student_essay>), and evidence record (<evidence_record>).\n"
        "{evidence_instruction}\n"
        "- You will reason carefully as to why you chose this score following the rubric and guidelines.\n"
        "- You will provide a detailed step-by-step explanation of your reasoning for the score.\n"
        "- You will provide feedback for the student on how to improve the vocabulary and sentence variety of their essay.\n"
        "- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student "
        "improve their score in future essays.\n\n"
        "The rubric or rubrics for this essay is as follows:\n"
        "<vocabulary_rubric>\n"
        "VOCABULARY_RUBRIC_PLACEHOLDER\n"
        "</vocabulary_rubric>\n\n"
        "The structured evidence record extracted from the student essay:\n"
        "<evidence_record>\n"
        "EVIDENCE_RECORD_PLACEHOLDER\n"
        "</evidence_record>\n\n"
        "The prompt is as follows:\n"
        "<essay_prompt>\n"
        "PROMPT_PLACEHOLDER\n"
        "</essay_prompt>\n\n"
        "Review the given rubric, evidence record and prompt carefully and score the <student_essay>.\n"
        "Provide a numerical score by using the provided rubric's guidance. The score should be a number between 0 and 6.\n"
        "Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student "
        "improve their score in future essays.\n\n"
        "OUTPUT_FORMAT_PLACEHOLDER\n"
    )

    grammar_system_prompt = (
        "You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric.\n"
        "You specialize in scoring the grammar and mechanics of an essay.\n"
        "Please provide a numerical score for the provided essay considering all aspects of the specified rubric.\n\n"
        "- Provide an appropriate holistic grammar score.\n"
        "- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.\n"
        "- You will carefully read the rubric (<grammar_rubric>), prompt (<essay_prompt>), "
        "student essay (<student_essay>), and evidence record (<evidence_record>).\n"
        "{evidence_instruction}\n"
        "- You will reason carefully as to why you chose this score following the rubric and guidelines.\n"
        "- You will provide a detailed step-by-step explanation of your reasoning for the score.\n"
        "- You will provide feedback for the student on how to improve the grammar and mechanics of their essay.\n"
        "- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student "
        "improve their score in future essays.\n\n"
        "The rubric or rubrics for this essay is as follows:\n"
        "<grammar_rubric>\n"
        "GRAMMAR_RUBRIC_PLACEHOLDER\n"
        "</grammar_rubric>\n\n"
        "The structured evidence record extracted from the student essay:\n"
        "<evidence_record>\n"
        "EVIDENCE_RECORD_PLACEHOLDER\n"
        "</evidence_record>\n\n"
        "The prompt is as follows:\n"
        "<essay_prompt>\n"
        "PROMPT_PLACEHOLDER\n"
        "</essay_prompt>\n\n"
        "Review the given rubric, evidence record and prompt carefully and score the <student_essay>.\n"
        "Provide a numerical score by using the provided rubric's guidance. The score should be a number between 0 and 6.\n"
        "Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student "
        "improve their score in future essays.\n\n"
        "OUTPUT_FORMAT_PLACEHOLDER\n"
    )

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "argumentative_system_prompt": cls.argumentative_system_prompt,
            "vocabulary_system_prompt":    cls.vocabulary_system_prompt,
            "grammar_system_prompt":       cls.grammar_system_prompt,
            "input_prompt":                cls.input_prompt,
            "aspect_rubrics":              cls.aspect_rubrics,
        }

    @classmethod
    def format_prompt_inference(
        cls,
        grading_instruction: dict,
        agent_rubric_type: str,
        current_aspect_rubric: str,
        evidence_json: str = "",   # ← NEW: 4th parameter (default="" for backwards compat)
    ) -> str:
        """
        Format the agent prompt with optional evidence JSON injected.

        Two-step approach to avoid placeholder conflicts:
          1. .format(evidence_instruction=...) — injects the evidence instruction text
          2. .replace() — injects rubric, evidence JSON, prompt, output format
             (avoids collisions with existing {}-style placeholders in base prompts)
        """
        essay_text   = grading_instruction["essay_text"]
        prompt_text  = grading_instruction["prompt"]
        ev_instruction = (
            cls._EVIDENCE_INSTRUCTION if evidence_json
            else cls._NO_EVIDENCE_INSTRUCTION
        )
        evidence_content = evidence_json if evidence_json else "No evidence record provided."

        if agent_rubric_type == "vocabulary":
            system_prompt = (
                cls.vocabulary_system_prompt
                .format(evidence_instruction=ev_instruction)
                .replace("VOCABULARY_RUBRIC_PLACEHOLDER", current_aspect_rubric)
                .replace("EVIDENCE_RECORD_PLACEHOLDER",   evidence_content)
                .replace("PROMPT_PLACEHOLDER",            prompt_text)
                .replace("OUTPUT_FORMAT_PLACEHOLDER",     cls.output_format)
            )

        elif agent_rubric_type == "grammar":
            system_prompt = (
                cls.grammar_system_prompt
                .format(evidence_instruction=ev_instruction)
                .replace("GRAMMAR_RUBRIC_PLACEHOLDER",    current_aspect_rubric)
                .replace("EVIDENCE_RECORD_PLACEHOLDER",   evidence_content)
                .replace("PROMPT_PLACEHOLDER",            prompt_text)
                .replace("OUTPUT_FORMAT_PLACEHOLDER",     cls.output_format)
            )

        else:  # argumentative (aspects 1, 2, 3)
            system_prompt = (
                cls.argumentative_system_prompt
                .format(evidence_instruction=ev_instruction)
                .replace("ARGUMENTATIVE_RUBRIC_PLACEHOLDER", current_aspect_rubric)
                .replace("EVIDENCE_RECORD_PLACEHOLDER",      evidence_content)
                .replace("PROMPT_PLACEHOLDER",               prompt_text)
                .replace("OUTPUT_FORMAT_PLACEHOLDER",        cls.output_format)
            )

        input_prompt_formatted = cls.input_prompt.format(essay_text=essay_text)

        return cls.alpaca_prompt.format(system_prompt, input_prompt_formatted, "")
