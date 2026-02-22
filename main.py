from resolve_state import resolve_state_code
from get_state_judges import get_state_judges
from get_district_judges import get_district_judges
from judge_probability import get_judge_pmf
from attorney_ranking import get_judge_attorney_rankings, get_weighted_attorney_rankings
from speech_input import capture_prompt
from narrate_results import build_narration_script, narrate

prompt = capture_prompt()
print(f"You said: {prompt}")

state_code = resolve_state_code(prompt)
state_judges = get_state_judges(state_code)
district_judges = get_district_judges(location_description=prompt, judges_list=state_judges)
judge_pmf = get_judge_pmf(district_judges)
judge_attorney_rankings = get_judge_attorney_rankings(district_judges)
attorney_rankings = get_weighted_attorney_rankings(judge_pmf, judge_attorney_rankings)

script = build_narration_script(
    prompt=prompt,
    district_judges=district_judges,
    judge_pmf=judge_pmf,
    judge_attorney_rankings=judge_attorney_rankings,
    attorney_rankings=attorney_rankings,
)
print(script)
narrate(script)








