class Config:
    OLLAMA_CONFIG = {
        "model": "qwen3:8b",
        "base_url": "http://localhost:11434",
        "temperature": 0.0
    }
    JSON_FINED_MODEL_PATH="D:/PythonProjects/AiTools/JSON模型微调/qwen_json_fixer_unsloth/merged_model"
    coarse_model_path = "D:/endedingModel/bge-base-zh-v1___5"   # 粗排模型
    fine_model_path = "D:/endedingModel/bge-m3"  # 精排模型（BGE-M3）
    device = "cuda"  # 或 "cpu"
    use_gpu = True
    PROJECT_ROOT_PATH = "D:/PythonProjects/AiTools/医疗Langgraph实验_aaaa/resources/prompt"

    PROMPT_PATHS = {
        "intent_sys": PROJECT_ROOT_PATH + "/prompt_medical_intent_info_sys.txt",
        "intent_user": PROJECT_ROOT_PATH + "/prompt_medical_intent_info_user.txt",
        "chat_intent_choice_sys": PROJECT_ROOT_PATH + "/chat/prompt_chat_intent_choice_sys.txt",
        "chat_intent_choice_user": PROJECT_ROOT_PATH + "/chat/prompt_chat_intent_choice_user.txt",
        "medication_keyword_sys": PROJECT_ROOT_PATH + "/chat/prompt_medication_keyword_sys.txt",
        "medication_keyword_user": PROJECT_ROOT_PATH + "/chat/prompt_medication_keyword_user.txt",
        "medical_chat_answer_sys": PROJECT_ROOT_PATH + "/chat/prompt_medical_chat_answer_sys.txt",
        "medical_chat_answer_user": PROJECT_ROOT_PATH + "/chat/prompt_medical_chat_answer_user.txt",
        "primitive_concept_sys": PROJECT_ROOT_PATH + "/prompt_primitive_concept_sys.txt",
        "primitive_concept_user": PROJECT_ROOT_PATH + "/prompt_primitive_concept_user.txt",
        "disease_course_sys": PROJECT_ROOT_PATH + "/prompt_tp_disease_course_sys.txt",
        "disease_course_user": PROJECT_ROOT_PATH + "/prompt_tp_disease_course_user.txt",
        "human_body_system_sys": PROJECT_ROOT_PATH + "/prompt_tp_human_body_system_sys.txt",
        "human_body_system_user": PROJECT_ROOT_PATH + "/prompt_tp_human_body_system_user.txt",
        "manifestation_characteristics_sys": PROJECT_ROOT_PATH + "/prompt_tp_manifestation_characteristics_sys.txt",
        "manifestation_characteristics_user": PROJECT_ROOT_PATH + "/prompt_tp_manifestation_characteristics_user.txt",
        "symptom_nature_sys": PROJECT_ROOT_PATH + "/prompt_tp_symptom_nature_sys.txt",
        "symptom_nature_user": PROJECT_ROOT_PATH + "/prompt_tp_symptom_nature_user.txt",
        "disease_severity_sys": PROJECT_ROOT_PATH + "/prompt_disease_severity_sys.txt",
        "disease_severity_user": PROJECT_ROOT_PATH + "/prompt_disease_severity_user.txt",
        "sufficiency_decision_sys": PROJECT_ROOT_PATH + "/discover/prompt_sufficiency_decision_sys.txt",
        "sufficiency_decision_user": PROJECT_ROOT_PATH + "/discover/prompt_sufficiency_decision_user.txt",
        "summary_disease_sys": PROJECT_ROOT_PATH + "/discover/prompt_summary_disease_sys.txt",
        "summary_disease_user": PROJECT_ROOT_PATH + "/discover/prompt_summary_disease_user.txt",
        "extract_disease_name_sys": PROJECT_ROOT_PATH + "/discover/prompt_extract_disease_name_sys.txt",
        "extract_disease_name_user": PROJECT_ROOT_PATH + "/discover/prompt_extract_disease_name_user.txt",
        "disease_analysis_sys": PROJECT_ROOT_PATH + "/discover/prompt_disease_analysis_sys.txt",
        "disease_analysis_user": PROJECT_ROOT_PATH + "/discover/prompt_disease_analysis_user.txt",
        "concomitant_symptoms_sys": PROJECT_ROOT_PATH + "/discover/prompt_concomitant_symptoms_sys.txt",
        "concomitant_symptoms_user": PROJECT_ROOT_PATH + "/discover/prompt_concomitant_symptoms_user.txt",
        "generate_solution_sys": PROJECT_ROOT_PATH + "/discover/prompt_generate_solution_sys.txt",
        "generate_solution_user": PROJECT_ROOT_PATH + "/discover/prompt_generate_solution_user.txt",
        "extract_disease_info_sys": PROJECT_ROOT_PATH + "/discover/prompt_extract_disease_info_sys.txt",
        "extract_disease_info_user": PROJECT_ROOT_PATH + "/discover/prompt_extract_disease_info_user.txt",
        "chief_complaint_sys": PROJECT_ROOT_PATH + "/discover/prompt_chief_complaint_sys.txt",
        "chief_complaint_user": PROJECT_ROOT_PATH + "/discover/prompt_chief_complaint_user.txt",
        "missing_questions_sys": PROJECT_ROOT_PATH + "/discover/prompt_missing_questions_sys.txt",
        "missing_questions_user": PROJECT_ROOT_PATH + "/discover/prompt_missing_questions_user.txt",
        "ask_question_by_symptoms_sys": PROJECT_ROOT_PATH + "/discover/prompt_ask_question_by_symptoms_sys.txt",
        "ask_question_by_symptoms_user": PROJECT_ROOT_PATH + "/discover/prompt_ask_question_by_symptoms_user.txt",
    }

    MEDICAL_CHAT_SESSION = "D:/PythonProjects/AiTools/医疗Langgraph实验_aaaa/data/chat_sessions"

    # NEO4J 配置
    NEO4J_URL = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"

config = Config()