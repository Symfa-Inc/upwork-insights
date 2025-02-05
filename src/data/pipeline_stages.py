from typing import List, Tuple, Type

from src.data.feature_processors import (
    BooleanProcessor,
    DoNothingProcessor,
    FillNaProcessor,
    FrequencyProcessor,
    ListProcessor,
    NumericProcessor,
    OneHotProcessor,
    OrdinalProcessor,
    TextProcessorClustering,
    TextProcessorPCA,
    TextProcessorStatistics,
)
from src.data.feature_processors.base_processor import BaseProcessor
from src.data.feature_processors.cosine_k_means import CosineKMeans
from src.data.feature_processors.ppa import PCAWithPreProcessing

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


def get_stages(model: str, embeddings_dir: str) -> List[Tuple[str, Type[BaseProcessor], dict]]:
    stages: List[Tuple[str, Type[BaseProcessor], dict]] = [
        ('id', DoNothingProcessor, {}),
        # Title
        (
            'title',
            TextProcessorPCA,
            {
                'pca_class': PCAWithPreProcessing,
                'pca_params': {'preprocessing_method': 'normalize'},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'title',
            TextProcessorClustering,
            {
                'clustering_class': CosineKMeans,
                'clustering_params': {'n_clusters': 7},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'title',
            TextProcessorStatistics,
            {
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        # Description
        (
            'description',
            TextProcessorPCA,
            {
                'pca_class': PCAWithPreProcessing,
                'pca_params': {'preprocessing_method': 'normalize'},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'description',
            TextProcessorClustering,
            {
                'clustering_class': CosineKMeans,
                'clustering_params': {'n_clusters': 7},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'description',
            TextProcessorStatistics,
            {
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        # ('publsih_time', DeleteProcessor, {}),
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
        ('english_skill', FillNaProcessor, {}),
        # ('english_proficiency', DeleteProcessor, {}),
        ('freelancer_type', FillNaProcessor, {}),
        ('is_rising_talent', BooleanProcessor, {}),
        ('earnings', FillNaProcessor, {}),
        ('geo_country_name', FrequencyProcessor, {}),
        ('geo_country_population', NumericProcessor, {}),
        ('geo_country_gdppc', NumericProcessor, {}),
        ('geo_city_name', FrequencyProcessor, {}),
        ('geo_city_population', NumericProcessor, {}),
        ('geo_city_agglomeration', OneHotProcessor, {'threshold': 0.63}),
        ('is_local_market', BooleanProcessor, {}),
        # ('wh_status', DeleteProcessor, {}),
        ('wh_total_hours', NumericProcessor, {}),
        # ('wh_feedback_score', DeleteProcessor, {}),
        # ('wh_feedback_to_client_score', DeleteProcessor, {}),
        ('wh_total_charge', NumericProcessor, {}),
        ('wh_hourly_rate', NumericProcessor, {}),
        ('wh_duration', NumericProcessor, {}),
        # Company Name
        (
            'company_name',
            TextProcessorPCA,
            {
                'pca_class': PCAWithPreProcessing,
                'pca_params': {'preprocessing_method': 'normalize'},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'company_name',
            TextProcessorClustering,
            {
                'clustering_class': CosineKMeans,
                'clustering_params': {'n_clusters': 20},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'company_name',
            TextProcessorStatistics,
            {
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        # Company Description
        (
            'company_description',
            TextProcessorPCA,
            {
                'pca_class': PCAWithPreProcessing,
                'pca_params': {'preprocessing_method': 'normalize'},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'company_description',
            TextProcessorClustering,
            {
                'clustering_class': CosineKMeans,
                'clustering_params': {'n_clusters': 7},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'company_description',
            TextProcessorStatistics,
            {
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        # Company Summary
        (
            'company_summary',
            TextProcessorPCA,
            {
                'pca_class': PCAWithPreProcessing,
                'pca_params': {'preprocessing_method': 'normalize'},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'company_summary',
            TextProcessorClustering,
            {
                'clustering_class': CosineKMeans,
                'clustering_params': {'n_clusters': 54},
                'drop_column': False,
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        (
            'company_summary',
            TextProcessorStatistics,
            {
                'model': model,
                'embeddings_dir': embeddings_dir,
            },
        ),
        # ('company_size', DeleteProcessor, {}),
        ('company_industry', FrequencyProcessor, {}),
        ('company_visible', BooleanProcessor, {}),
        ('company_jobs_posted_count', NumericProcessor, {}),
        ('company_jobs_filled_count', NumericProcessor, {}),
        ('company_feedback_count', NumericProcessor, {}),
        ('company_hours_count', NumericProcessor, {}),
        ('company_total_charges_amount', NumericProcessor, {}),
        ('company_feedback_score', NumericProcessor, {}),
        ('company_avg_hourly_rate', NumericProcessor, {}),
        (
            'company_css_tier',
            FillNaProcessor,
            {},
        ),  # Does not change anything. Fills NA with mode value
        ('company_hire_rate', DoNothingProcessor, {}),
        ('company_experience', NumericProcessor, {}),
        ('segmentation_data_value', OneHotProcessor, {}),
        ('segmentation_data_label', OneHotProcessor, {}),
        ('engagement_duration', OrdinalProcessor, {'mapping': ENGAGEMENT_DURATION_MAPPING}),
        ('engagement_type', OneHotProcessor, {}),
        ('budget_min', NumericProcessor, {}),
        ('budget_max', NumericProcessor, {}),
    ]
    return stages
