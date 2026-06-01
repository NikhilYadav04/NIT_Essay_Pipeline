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
Score 3: The essay contains contains occasional major errors or frequent minor errors in grammar, usage, or mechanics that can interfere with meaning.
Score 2: The essay contains serious errors in grammar, usage, or mechanics that frequently obscure meaning.
Score 1: The essay contains pervasive errors in grammar, usage, or mechanics that result in incoherence.
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""

    aspect_rubrics = [("argumentative", aspect_1_rubric, "Aspect 1: Quality of the response to the prompt instructions"), ("argumentative", aspect_2_rubric, "Aspect 2: Considering the complexities of the issue"), ("argumentative", aspect_3_rubric, "Aspect 3: Organizing, developing, and expressing ideas"), ("vocabulary", aspect_4_rubric, "Aspect 4: Vocabulary and sentence variety"), ("grammar", aspect_5_rubric, "Aspect 5: Grammar and mechanics")]


    argumentative_system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
You specialize in scoring the argumentative qualities of an essay.
Please provide a numerical score for the provided essay considering all aspects of the specified rubric.


- Provide an appropriate holistic argumentative score.
- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.
- You will carefully read the rubric (<argumentative_rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed step-by-step explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve the argumentative qualities of their essay.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

The rubric or rubrics for this essay is as follows:
<argumentative_rubric>
{argumentative_rubric}
</argumentative_rubric>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given rubric and prompt carefully and score the <student_essay>.
Provide a numerical score by using the provided rubric's guidance. The score should be a number between 0 and 6.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""
    vocabulary_system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
You specialize in scoring the vocabulary and sentence variety of an essay.
Please provide a numerical score for the provided essay considering all aspects of the specified rubric.


- Provide an appropriate holistic vocabulary score.
- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.
- You will carefully read the rubric (<vocabulary_rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed step-by-step explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve the vocabulary and sentence variety of their essay.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

The rubric or rubrics for this essay is as follows:
<vocabulary_rubric>
{vocabulary_rubric}
</vocabulary_rubric>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given rubric and prompt carefully and score the <student_essay>.
Provide a numerical score by using the provided rubric's guidance. The score should be a number between 0 and 6.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""

    grammar_system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
You specialize in scoring the grammar and mechanics of an essay.
Please provide a numerical score for the provided essay considering all aspects of the specified rubric.


- Provide an appropriate holistic grammar score.
- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.
- You will carefully read the rubric (<grammar_rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed step-by-step explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve the grammar and mechanics of their essay.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

The rubric or rubrics for this essay is as follows:
<grammar_rubric>
{grammar_rubric}
</grammar_rubric>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given rubric and prompt carefully and score the <student_essay>.
Provide a numerical score by using the provided rubric's guidance. The score should be a number between 0 and 6.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""
    
    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "argumentative_system_prompt": cls.argumentative_system_prompt,
            "vocabulary_system_prompt": cls.vocabulary_system_prompt,
            "grammar_system_prompt": cls.grammar_system_prompt,
            "input_prompt": cls.input_prompt,
            "aspect_rubrics": cls.aspect_rubrics,
        }

    @classmethod
    def format_prompt_inference(cls, grading_instruction: dict, agent_rubric_type: str, current_aspect_rubric: str) -> str:
        essay_text = grading_instruction["essay_text"]
        if agent_rubric_type == "vocabulary":
            system_prompt_formatted = cls.vocabulary_system_prompt.format(
                vocabulary_rubric=current_aspect_rubric,
                prompt=grading_instruction["prompt"],
                output_format=cls.output_format,
                min_score=0,
                max_score=6
            )
        elif agent_rubric_type == "grammar":
            system_prompt_formatted = cls.grammar_system_prompt.format(
                grammar_rubric=current_aspect_rubric,
                prompt=grading_instruction["prompt"],
                output_format=cls.output_format,
                min_score=0,
                max_score=6
            )
        else:
            system_prompt_formatted = cls.argumentative_system_prompt.format(
                argumentative_rubric=current_aspect_rubric,
                prompt=grading_instruction["prompt"],
                output_format=cls.output_format,
                min_score=0,
                max_score=6
            )
        input_prompt_formatted = cls.input_prompt.format(
            essay_text=essay_text
        )

        return cls.alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")
