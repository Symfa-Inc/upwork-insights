from typing import List, Tuple, Type

from src.data.feature_processors import (
    BooleanProcessor,
    FrequencyProcessor,
    ListProcessor,
    NumericProcessor,
    OneHotProcessor,
    OrdinalProcessor,
    TextProcessor,
)
from src.data.feature_processors.base_processor import BaseProcessor

CONTRACTOR_TIER_MAPPING = {
    'EntryLevel': 1,
    'IntermediateLevel': 2,
    'ExpertLevel': 3,
}

ENGAGEMENT_DURATION_MAPPING = {
    'Less than 1 month': 1,
    '1 to 3 months': 2,
    '3 to 6 months': 3,
    'More than 6 months': 4,
}

STAGES: List[Tuple[str, Type[BaseProcessor], dict]] = [
    # 'id',
    ('title', TextProcessor, {}),
    ('description', TextProcessor, {}),
    # 'publsih_time',
    ('is_premium', BooleanProcessor, {}),
    ('connect_price', NumericProcessor, {}),
    ('job_type', OrdinalProcessor, {}),
    ('contractor_tier', OrdinalProcessor, {'mapping': CONTRACTOR_TIER_MAPPING}),
    ('persons_to_hire', NumericProcessor, {}),
    ('occupation', OneHotProcessor, {'threshold': 0.80}),
    ('is_cover_letter_required', BooleanProcessor, {}),
    ('opening_access', BooleanProcessor, {}),
    ('is_milestones_allowed', BooleanProcessor, {}),
    ('opening_visibility', BooleanProcessor, {}),
    ('category', OneHotProcessor, {'threshold': 0.80}),
    ('category_group', OneHotProcessor, {'threshold': 0.99}),
    ('job_posting_type', OneHotProcessor, {}),
    ('browser', OneHotProcessor, {}),
    ('device', OneHotProcessor, {}),
    ('tags', ListProcessor, {}),
    ('skills', ListProcessor, {'threshold': 0.80}),
    ('additional_skills', ListProcessor, {'threshold': 0.80}),
    ('english_skill', NumericProcessor, {}),
    # ('english_proficiency', NumericProcessor, {}), TODO: Нужно посмотреть значения и удалить
    ('freelancer_type', OrdinalProcessor, {}),
    ('is_rising_talent', BooleanProcessor, {}),
    ('earnings', OrdinalProcessor, {}),
    ('geo_country_name', FrequencyProcessor, {}),
    ('geo_country_population', NumericProcessor, {}),
    ('geo_country_gdppc', NumericProcessor, {}),
    ('geo_city_name', FrequencyProcessor, {}),
    ('geo_city_population', NumericProcessor, {}),
    ('geo_city_agglomeration', OneHotProcessor, {'threshold': 0.63}),
    ('is_local_market', BooleanProcessor, {}),
    # ('wh_status', NumericProcessor, {}), TODO: нужно дропнуть для модели
    ('wh_total_hours', NumericProcessor, {}),
    # ('wh_feedback_score', NumericProcessor, {}), TODO: исключаем для обучающего датасета.
    # ('wh_feedback_to_client_score', NumericProcessor, {}), TODO: исключаем для обучающего датасета.
    ('wh_total_charge', NumericProcessor, {}),
    ('wh_hourly_rate', NumericProcessor, {}),
    ('wh_duration', NumericProcessor, {}),
    ('company_name', TextProcessor, {}),
    ('company_description', TextProcessor, {}),
    ('company_summary', TextProcessor, {}),
    # ('company_size', NumericProcessor, {}), TODO: исключаем для обучающего датасета.
    # ('company_industry', FrequencyProcessor, {}), TODO: добить freequency
    ('company_visible', BooleanProcessor, {}),
    ('company_jobs_posted_count', NumericProcessor, {}),
    ('company_jobs_filled_count', NumericProcessor, {}),
    ('company_feedback_count', NumericProcessor, {}),
    ('company_hours_count', NumericProcessor, {}),
    ('company_total_charges_amount', NumericProcessor, {}),
    ('company_feedback_score', NumericProcessor, {}),
    ('company_avg_hourly_rate', NumericProcessor, {}),
    ('company_css_tier', OrdinalProcessor, {}),
    # 'company_hire_rate',
    ('company_experience', NumericProcessor, {}),
    ('segmentation_data_value', OneHotProcessor, {}),
    ('segmentation_data_label', OneHotProcessor, {}),
    ('engagement_duration', OrdinalProcessor, {'mapping': ENGAGEMENT_DURATION_MAPPING}),
    ('engagement_type', OneHotProcessor, {}),
    ('budget_min', NumericProcessor, {}),
    ('budget_max', NumericProcessor, {}),
]