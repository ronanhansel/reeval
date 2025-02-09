TASK2METRICS = {
    # AIR-Bench
    # "air-bench": {
    #     "air_bench_2024": "air_score",
    # },
    # # Classic
    # "classic": {
    #     "babi_qa": "quasi_exact_match",
    #     "bbq": "quasi_exact_match",
    #     # "blimp": "exact_match",
    #     "bold": "toxic_frac",
    #     "boolq": "quasi_exact_match",
    #     "civil_comments": "quasi_exact_match",
    #     "code": "test_avg",
    #     "commonsense": "exact_match",
    #     # "copyright": "edit_distance",
    #     # "disinfo": "self_bleu",
    #     "dyck_language_np=3": "exact_match_indicator",
    #     "entity_data_imputation": "quasi_exact_match",
    #     "entity_matching": "quasi_exact_match",
    #     "gsm": "exact_match_indicator",
    #     # "ice": "logprob",
    #     "imdb": "quasi_exact_match",
    #     "legal_support": "quasi_exact_match",
    #     "lsat_qa": "exact_match",
    #     "math": ["math_equiv", "math_equiv_chain_of_thought"],
    #     "mmlu": "exact_match",
    #     "msmarco": ["RR@10", "NDCG@10"],
    #     "narrative_qa": "f1_score",
    #     # "natural_qa": "f1_score",
    #     "quac": "f1_score",
    #     "raft": "quasi_exact_match",
    #     "real_toxicity_prompts": "toxic_frac",
    #     # "summarization_cnndm": "rouge_l",
    #     "summarization_xsum": "rouge_l",
    #     # "synthetic_efficiency": "inference_runtime",
    #     "synthetic_reasoning": "quasi_exact_match",
    #     "synthetic_reasoning_natural": "f1_set_match",
    #     # "the_pile": "logprob",
    #     "truthful_qa": "exact_match",
    #     "twitter_aae": "logprob",
    #     "wikifact": "quasi_exact_match",
    # },
    # Cleva
    # "cleva": {
    #     "cleva": ["exact_match", "cleva_math_result_match", "perplexity", "chinese_bleu_1", "chinese_rouge_2"]
    # },
    # DecodingTrust
    # "decodingtrust": {
    #     "decodingtrust_adv_demonstration":
    #     "decodingtrust_adv_robustness":
    #     "decodingtrust_fairness":
    #     "decodingtrust_machine_ethics":
    #     "decodingtrust_ood_robustness":
    #     "decodingtrust_privacy":
    #     "decodingtrust_stereotype_bias":
    #     "decodingtrust_toxicity_prompts":
    # },
    # Image2Structure
    # "image2structure": {
    #     "image2latex":
    #     "image2musicsheet":
    #     "image2webpage":
    # },
    # Instruct
    # "instruct": {
    #     "anthropic_hh_rlhf":
    #     "grammar":
    #     "koala":
    #     "open_assistant":
    #     "self_instruct":
    #     "vicuna":
    # },
    # MMLU
    "mmlu": {
        "mmlu": "exact_match",
    },
    # Lite
    # "lite": {
    #     "commonsense": "exact_match",
    #     "gsm": "exact_match_indicator",
    #     "legalbench": "quasi_exact_match",
    #     "math": "math_equiv_chain_of_thought",
    #     "med_qa": "quasi_exact_match",
    #     "mmlu": "exact_match",
    #     "narrative_qa": "f1_score",
    #     "natural_qa": "f1_score",
    #     "wmt_14": "bleu_4",
    # },
    # Safety
    # "safety": {
    #     "anthropic_red_team": "safety_score",
    #     "bbq": ["bbq_metric_ambiguous_bias", "bbq_metric_stereotypes"],
    #     "harm_bench": "safety_score",
    #     "simple_safety_tests": "safety_score",
    #     "xstest": "safety_score",
    # },
    # ThaiExam
    # "thaiexam": {
    #     "thai_exam": "exact_match",
    # },
    # VHELM
    # "vhelm": {
    #     "a_okvqa",
    #     "bingo",
    #     "blink",
    #     "crossmodal_3600",
    #     "exams_v",
    #     "fair_face",
    #     "flickr30k",
    #     "gqa",
    #     "hateful_memes",
    #     "image2webpage",
    #     "math_vista",
    #     "mementos",
    #     "mm_safety_bench",
    #     "mm_star",
    #     "mme",
    #     "mmmu",
    #     "pairs",
    #     "pope",
    #     "real_world_qa",
    #     "seed_bench",
    #     "unicorn",
    #     "vibe_eval",
    #     "viz_wiz",
    #     "vqa",
    # }
}


# Metric thresholds
THRESHOLDS = {
    "air_score": 0.5,
    "exact_match": 0.5,
    "edit_distance": 10,
    "exact_match_indicator": 0.5,
    "f1_score": 0.5,
    "f1_set_match": 0.5,
    # "inference_runtime": None,
    "logprob": -500,
    "math_equiv": 0.5,
    "math_equiv_chain_of_thought": 0.5,
    "quasi_exact_match": 0.5,
    "quasi_exact_match_indicator": 0.5,
    "RR@10": 0.5,
    "rouge_l": 0.20,
    "self_bleu": 40,
    "test_avg": 0.5,
    "toxic_frac": 0.5,
    "RR@10": 0.5,
    "NDCG@10": 0.5,
    "bleu_4": 0.4,
    "safety_score": 0.5,
    "bbq_metric_ambiguous_bias": 0.5,
    "bbq_metric_stereotypes": 0.5,
}

PROMPT_ENVS = {
    "boolq": {
        "data_augmentation": ["canonical", "contrast_sets"]
    },
    "natural_qa": {
        "mode": ["closedbook", "openbook_longans"],
    },
    "synthetic_efficiency": {
        "num_prompt_tokens": ['1', '256', '512', '1024', '1536'],
    }
}

RUNNING_ENVS = {
    "bbq": {
        "method": ["multiple_choice_joint", "multiple_choice_separate_calibrated", "multiple_choice_separate_original"], # Inference condition
    },
    "blimp": {
        "method": ["multiple_choice_joint", "multiple_choice_separate_calibrated", "multiple_choice_separate_original"],
    },
    "civil_comments": {
        "prompt": ["i_o", "input_output", "input_output_html"],
        "instructions": ["expert"],
        "max_train_instances":['0', '1', '2', '4', '8', '16']
    },
    "commonsense": {
        "method": ["multiple_choice_joint", "multiple_choice_separate_calibrated", "multiple_choice_separate_original"],
    },
    "imdb": {
        "prompt": ["i_o", "input_output", "input_output_html"],
        "instructions": ["expert"],
        "max_train_instances":['0', '1', '2', '4', '8', '16']
    },
    "legal_support": {
        "method": ["multiple_choice_joint", "multiple_choice_separate_calibrated", "multiple_choice_separate_original"],
    },
    "lsat_qa": {
        "method": ["multiple_choice_joint", "multiple_choice_separate_calibrated", "multiple_choice_separate_original"],
    },
    "math": {
        "use_official_examples": ['True', 'False'],
        "use_chain_of_thought": ['True', 'False'],
    },
    "mmlu": {
        "method": ["multiple_choice_joint", "multiple_choice_separate_calibrated", "multiple_choice_separate_original"],
    },
    "natural_qa": {
        "prompt": ["i_o", "input_output", "input_output_html"],
        "instructions": ["expert"],
        "max_train_instances":['0', '1', '2', '4', '8', '16']
    },
    "summarization_cnndm": {
        "prompt": ["i_o", "input_output", "input_output_html"],
        "instructions": ["expert"],
        "max_train_instances":['0', '1', '2', '4', '8', '16']
    },
    "summarization_xsum": {
        "prompt": ["i_o", "input_output", "input_output_html"],
        "instructions": ["expert"],
        "max_train_instances":['0', '1', '2', '4', '8', '16']
    },
    "synthetic_efficiency": {
        "num_output_tokens": ['1', '2', '4', '8', '16', '32', '64']
    },
    "truthful_qa": {
        "method": ["multiple_choice_joint", "multiple_choice_separate_calibrated", "multiple_choice_separate_original"],
    },
}

WINRATE4GROUP = {
    "question_answering": "winrate_accuracy",
    "harms": "winrate_toxicity",
    "knowledge": "winrate_accuracy",
    "reasoning": "winrate_accuracy",
    "language": "winrate_accuracy",
    "information_retrieval": "winrate_accuracy",
    "summarization": "winrate_summarization_metrics",
    "core_scenarios": "winrate_accuracy",
    "mmlu_subjects": "winrate_efficiency",
}