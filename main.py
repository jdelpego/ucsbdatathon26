from resolve_state import resolve_state_code
from get_state_judges import get_state_judges
from get_district_judges import get_district_judges
from judge_probability import get_judge_pmf

prompt = "I need help with my court case in San Diego"

state_code = resolve_state_code(prompt)
state_judges = get_state_judges(state_code)
district_judges = get_district_judges(location_description=prompt, judges_list=state_judges)
judge_pmf = get_judge_pmf(district_judges)





