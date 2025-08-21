import joblib
import pandas as pd
import numpy as np
import shap
import uvicorn
import requests
import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from preprocessing import load_and_prepare_lookup_tables, generate_features

# 기본 설정
app = FastAPI(title="전세 보증사고 위험 예측 통합 API")
LOOKUP_TABLES = load_and_prepare_lookup_tables()
try:
    model = joblib.load('./model/xgb_model.pkl')
    model_features = model.get_booster().feature_names
    explainer = shap.TreeExplainer(model)
    print("✅ 모델과 SHAP 설명자를 성공적으로 로드했습니다.")
except Exception as e:
    print(f"❌ 모델 로딩 중 오류 발생: {e}")
    model, explainer = None, None

# 환경 변수에서 외부 AI 서버 주소 불러오기
EXTERNAL_AI_URL = os.getenv("EXTERNAL_AI_URL")

# Pydantic 데이터 모델 정의
class ContractInput(BaseModel):
    보증시작월: int
    보증완료월: int
    주택가액: float
    임대보증금액: float
    선순위: float
    시도: str
    주택구분: str

class FinalPredictionResult(BaseModel):
    risk_score: float
    risk_level: str
    explanation: List[Dict[str, Any]]
    ai_explanation: str
    original_input: Dict[str, Any]

# 핵심 예측 로직
def _calculate_prediction(features: ContractInput) -> Dict[str, Any]:
    if not model:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    
    input_df = generate_features(features.dict(), LOOKUP_TABLES)
    final_features_order = [col.replace('(%)', 'pct') for col in model_features]
    
    try: 
        model_input_df = input_df[final_features_order]
    except KeyError as e: 
        raise HTTPException(status_code=400, detail=f"모델 학습에 사용된 컬럼이 생성되지 않았습니다: {e}")

    risk_proba = model.predict_proba(model_input_df)[0, 1]
    shap_values = explainer.shap_values(model_input_df)

    explanation = []
    top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:3]
    for i in top_indices:
        feature_name = str(final_features_order[i])
        explanation.append({
            "feature": feature_name,
            "value": model_input_df.iloc[0, i].item(),
            "contribution": round(shap_values[0, i].item(), 4),
            "description": "위험도를 높이는 주요 원인입니다." if shap_values[0, i] > 0 else "위험도를 낮추는 요인입니다."
        })

    risk_level = "매우 높음" if risk_proba > 0.75 else "높음" if risk_proba > 0.5 else "보통"
    
    return {
        "risk_score": round(float(risk_proba), 4),
        "risk_level": risk_level,
        "explanation": explanation,
        "original_input": features.dict() 
    }

# API 엔드포인트
@app.post("/predict_and_explain", response_model=FinalPredictionResult)
def predict_and_explain_risk(features: ContractInput):

    prediction_result = _calculate_prediction(features)
    
    ai_explanation_text = "AI 설명 서버에서 답변을 받아오는 데 실패했습니다."
    try:
        # prediction_result 딕셔너리를 JSON 형식의 '문자열'로 변환합니다.
        prediction_json_string = json.dumps(prediction_result, ensure_ascii=False)
        
        # 외부 서버가 요구하는 형식에 맞게 '문자열'을 "question" 키로 감싸줍니다.
        ask_payload = {"question": prediction_json_string}
        print(prediction_json_string)
        # 외부 AI 서버에 JSON 데이터를 전송합니다.
        ask_response = requests.post(EXTERNAL_AI_URL, json=ask_payload, timeout=20)
        
        if ask_response.status_code == 200:
            ai_explanation_text = ask_response.json().get("answer", "AI 응답에서 'answer' 키를 찾을 수 없습니다.")
        else:
            ai_explanation_text = f"AI 설명 서버 에러: Status Code {ask_response.status_code}, Response: {ask_response.text}"

    except requests.exceptions.RequestException as e:
        ai_explanation_text = f"AI 설명 서버 연결 중 오류 발생: {e}"

    final_result = {
        **prediction_result,
        "ai_explanation": ai_explanation_text
    }
    
    return final_result
