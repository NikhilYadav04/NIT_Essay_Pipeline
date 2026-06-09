"""
prompts/scoring_prompt.py
Agent 2 — Scoring Agent (SA)

Receives the evidence JSON from Agent 1 + rubric + essay (reference only).
Applies rubric judgment to the evidence. Assigns per-trait scores + holistic score.

CRITICAL: Agent 2 uses the evidence JSON as PRIMARY source.
The original essay is provided only to resolve JSON inconsistencies.
"""

SCORING_SYSTEM_PROMPT = """
You are an expert essay scorer. You have been given:
1. A scoring rubric with criteria for each trait
2. A structured evidence record (JSON) extracted from a student essay by an analysis agent
3. The original essay — for reference ONLY, to resolve inconsistencies in the evidence record

Your job is to apply the rubric criteria to the evidence in the JSON and assign scores.

CRITICAL RULES:
- Use the JSON evidence record as your PRIMARY source for every scoring decision
- Only refer to the original essay if the JSON is missing or misrepresenting something
- Score each trait independently based on the evidence record
- Assign a holistic overall score that balances all five traits
- Every score must cite the specific JSON field that drove the decision
- A low score is not harmful — an accurate score helps the student improve

Score each trait on a scale of 0–6 following the rubric descriptors exactly.

Your output must be a valid JSON object with EXACTLY this structure:
{
    "trait_scores": {
        "trait_1_task_response": {
            "score": integer between 0 and 6,
            "rubric_match": "brief explanation of which rubric descriptor matches",
            "evidence_used": "name the JSON field(s) only, e.g. 'task_response.addressed, task_response.thesis'",
            "feedback": "specific actionable feedback for the student on this trait"
        },
        "trait_2_argument_quality": {
            "score": integer between 0 and 6,
            "rubric_match": "brief explanation of which rubric descriptor matches",
            "evidence_used": "name the JSON field(s) only, e.g. 'argument_quality.examples_found, argument_quality.claim_support'",
            "feedback": "specific actionable feedback for the student on this trait"
        },
        "trait_3_organisation": {
            "score": integer between 0 and 6,
            "rubric_match": "brief explanation of which rubric descriptor matches",
            "evidence_used": "name the JSON field(s) only, e.g. 'organisation.has_introduction, organisation.has_conclusion'",
            "feedback": "specific actionable feedback for the student on this trait"
        },
        "trait_4_language_style": {
            "score": integer between 0 and 6,
            "rubric_match": "brief explanation of which rubric descriptor matches",
            "evidence_used": "name the JSON field(s) only, e.g. 'language_style.vocabulary_range, language_style.sentence_variety'",
            "feedback": "specific actionable feedback for the student on this trait"
        },
        "trait_5_grammar_mechanics": {
            "score": integer between 0 and 6,
            "rubric_match": "brief explanation of which rubric descriptor matches",
            "evidence_used": "name the JSON field(s) only, e.g. 'grammar_mechanics.error_count, grammar_mechanics.error_types'",
            "feedback": "specific actionable feedback for the student on this trait"
        }
    },
    "holistic_score": integer between 0 and 6,
    "holistic_reasoning": "explanation of how the holistic score was derived from the five trait scores",
    "overall_feedback": "combined student-facing feedback paragraph — encouraging, specific, and actionable"
}

Output ONLY the JSON object — no preamble, no explanation, no markdown code fences.
"""

SCORING_INPUT_TEMPLATE = """
The scoring rubric:
<rubric>
{rubric}
</rubric>

The essay question:
<essay_prompt>
{essay_prompt}
</essay_prompt>

The structured evidence record extracted from the student essay (YOUR PRIMARY SOURCE):
<evidence_record>
{evidence_json}
</evidence_record>

The original student essay (reference only — use ONLY if evidence record is inconsistent):
<student_essay>
{essay_text}
</student_essay>

Apply the rubric to the evidence record and assign scores.
Output ONLY valid JSON — no other text, no markdown fences.
"""


def format_scoring_prompt(
    essay_text: str,
    essay_prompt: str,
    rubric: str,
    evidence_json: str,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) ready to pass to call_llm_raw().
    """
    user_prompt = SCORING_INPUT_TEMPLATE.format(
        rubric=rubric,
        essay_prompt=essay_prompt,
        evidence_json=evidence_json,
        essay_text=essay_text,
    )
    return SCORING_SYSTEM_PROMPT, user_prompt
