# LR Classification Experiment

Logistic Regression을 사용하여 LLM의 hidden state가 gambling 결과(bankruptcy vs voluntary_stop)를 예측할 수 있는지 분석합니다.

## 핵심 질문

> **모델의 internal activation만으로 bankruptcy vs safe를 구분할 수 있는가?**

## 실험 설계

### 데이터 소스
- 기존 slot machine 실험 결과 (Gemma, LLaMA)
- 새로운 대화 생성 없이 기존 데이터의 프롬프트 재구성

### 세 가지 분석 옵션

| Option | 설명 | 샘플 수 |
|--------|------|---------|
| **A** | 게임 시작 시점 (history 없음) | ~3,200 |
| **B** | 게임 종료 직전 (핵심) | ~3,200 |
| **C** | 모든 라운드 (trajectory) | ~32,000 |

### 파이프라인

```
JSON 데이터 → 프롬프트 재구성 → 모델 forward pass → hidden state → LR 분류
```

## 사용법

```bash
# 전체 실험 (Option B, 핵심)
python run_experiment.py --model gemma --option B --gpu 0

# 모든 옵션 실행
python run_experiment.py --model gemma --option all --gpu 0

# 두 모델 모두
python run_experiment.py --model all --option B --gpu 0
```

## 폴더 구조

```
lr_classification_experiment/
├── README.md
├── config.yaml
├── run_experiment.py          # 메인 실행 스크립트
└── src/
    ├── __init__.py
    ├── prompt_reconstruction.py  # 프롬프트 재구성
    ├── hidden_state_extractor.py # Hidden state 추출
    └── lr_classifier.py          # LR 분류 실험
```

## 예상 결과

- Option A: ~50-55% (chance level, 동일 입력)
- Option B: ~70-85% (컨텍스트 정보 포함)
- Option C: trajectory 분석 가능
