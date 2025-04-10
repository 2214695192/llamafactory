from typing import Dict, Any, List

# Neo4j 配置
NEO4J_CONFIG: Dict[str, Any] = {
    "uri": "bolt://10.26.2.130:7687",  # 服务器 Bolt 端口
    "user": "neo4j",                    # 用户名
    "password": "Zq08171911"           # 密码
}

# 硅基流动配置
SILICON_CONFIG: Dict[str, Any] = {
    "api_base": "https://api.siliconflow.cn/v1/chat/completions",  # 硅基流动 API 地址
    "temperature": 0.0,
    "max_tokens": 4000  # 增加 token 数以处理更长的文档
}

# 眼科知识图谱核心配置
KNOWLEDGE_GRAPH_CONFIG: Dict[str, Any] = {
    # 节点类型 (包含12个基础类型+8个眼科特化类型)
    "node_types": [
        # 基础医学类型
        "Disease", "Symptom", "Diagnosis", "Treatment", 
        "RiskFactor", "Prevention", "Complication", "Anatomy",
        "Medication", "MedicalDevice", "Pathogen", "Biomarker",
        
        # 眼科特化类型
        "Ocular_Structure",        # 眼部结构 (取代通用Anatomy)
        "Refractive_Status",       # 屈光状态
        "Ocular_Physiology",       # 眼生理指标
        "Ophthalmic_Imaging",      # 眼科影像检查
        "Laser_Procedure",         # 激光治疗
        "Intraocular_Lens",        # 人工晶体
        "Visual_Acuity",           # 视力
        "Visual_Field",            # 视野
        
        # 补充眼科疾病和手术类型
        "Retinal_Disease",         # 视网膜疾病
        "Glaucoma",                # 青光眼
        "Refractive_Error",        # 屈光不正
        "Cataract_Surgery",        # 白内障手术
        "Corneal_Transplant",      # 角膜移植
        "Photocoagulation",        # 光凝术
        "Laser_Trabeculoplasty",   # 激光小梁成形术
        "PRK",                     # PRK手术
        "LASIK",                   # LASIK手术
        "Monofocal_IOL",          # 单焦点人工晶状体
        "Multifocal_IOL",         # 多焦点人工晶状体
        "Toric_IOL"               # 散光矫正人工晶状体
    ],
    
    # 关系类型 (17基础+9眼科特化)
    "relationship_types": [
        # 基础医学关系
        "has_symptom", "diagnosed_by", "treated_by", 
        "has_risk_factor", "prevented_by", "leads_to",
        "part_of", "type_of", "stage_of", "causes",
        "associated_with", "measured_by", "located_in",
        "indicated_for", "contraindicates", "interacts_with",
        "classified_as",
        
        # 眼科特化关系
        "treated_with_laser",     # 激光治疗
        "graded_by",              # 分级标准
        "affects_region",         # 累及区域
        "risk_quantified_by",     # 量化风险指标
        "postoperative_complication_of",  # 术后并发症
        "refractive_correction_for",      # 屈光矫正适用
        "implanted_with",         # 植入物
        "visual_outcome_of",      # 视觉结果来源
        "screened_by"             # 筛查手段
    ],
    
    # 同义词映射表 (SNOMED CT标准)
    "synonym_mapping": {
        "Cataract": ["Lens_opacity", "Phacoscotasmus"],
        "Glaucoma": ["Optic_neuropathy", "Trabecular_blockage"],
        "LASIK": ["Laser_assisted_in_situ_keratomileusis", "Refractive_laser_surgery"],
        "IOP": ["Intraocular_pressure", "Ocular_tension"]
    },
    
    # 动态关系约束
    "relation_constraints": {
        "treated_with_laser": {
            "allowed_subjects": ["Retinal_Disease", "Glaucoma", "Refractive_Error"],
            "allowed_objects": ["Photocoagulation", "Laser_Trabeculoplasty", "PRK", "LASIK"]
        },
        "implanted_with": {
            "allowed_subjects": ["Cataract_Surgery", "Corneal_Transplant"],
            "allowed_objects": ["Monofocal_IOL", "Multifocal_IOL", "Toric_IOL"]
        },
        "postoperative_complication_of": {
            "required_properties": ["time_interval"]  # 必须包含时间属性
        }
    },
    
    # 分层分类体系
    "hierarchy": {
        "Ocular_Structure": {
            "Anterior_Segment": ["Cornea", "Iris", "Lens"],
            "Posterior_Segment": ["Retina", "Choroid", "Optic_nerve"],
            "Adnexa": ["Eyelid", "Lacrimal_gland"]
        },
        "Laser_Procedure": {
            "Therapeutic": ["Photocoagulation", "Iridotomy"],
            "Refractive": ["LASIK", "SMILE", "PRK"],
            "Diagnostic": ["OCT_scan", "Confocal_microscopy"]
        },
        "Refractive_Status": {
            "Myopia": ["Low_myopia", "High_myopia"],
            "Hyperopia": ["Axial_hyperopia", "Refractive_hyperopia"],
            "Astigmatism": ["Regular", "Irregular"]
        }
    },
    
    # 眼科专有属性
    "ocular_properties": {
        "Quantitative": ["Axial_length(mm)", "Corneal_thickness(μm)", "IOP(mmHg)"],
        "Qualitative": ["Optic_nerve_cupping", "Macular_edema_grade", "Lens_opacity_type"],
        "Temporal": ["Onset_age", "Postoperative_days"]
    }
}

def validate_config(config: Dict[str, Any]) -> bool:
    """确保配置符合眼科知识图谱规范"""
    required_sections = [
        'node_types', 'relationship_types', 
        'synonym_mapping', 'relation_constraints'
    ]
    
    # 检查必要模块
    for section in required_sections:
        if section not in config:
            raise ValueError(f"缺失关键配置模块: {section}")
    
    # 验证节点类型唯一性
    if len(config['node_types']) != len(set(config['node_types'])):
        duplicates = [item for item in set(config['node_types']) 
                    if config['node_types'].count(item) > 1]
        raise ValueError(f"重复的节点类型: {duplicates}")
    
    # 验证关系约束有效性
    for rel, constraints in config['relation_constraints'].items():
        if rel not in config['relationship_types']:
            raise ValueError(f"未定义的关系类型被约束: {rel}")
        if 'allowed_subjects' in constraints:
            for subj in constraints['allowed_subjects']:
                if subj not in config['node_types']:
                    raise ValueError(f"约束中使用了未定义的节点类型: {subj}")
    
    return True

# 初始化验证
if __name__ == "__main__":
    try:
        validate_config(KNOWLEDGE_GRAPH_CONFIG)
        print("✅ 配置验证通过")
    except Exception as e:
        print(f"❌ 配置错误: {str(e)}") 