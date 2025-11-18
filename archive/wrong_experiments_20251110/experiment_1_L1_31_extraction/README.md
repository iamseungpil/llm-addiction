# Experiment 1: L1-31 SAE Feature Extraction

## 현재 실행 중인 실험

**상태**: 🔄 진행 중 (Batch 3/3)
**시작**: 2025-11-10 18:01
**예상 완료**: 2025-11-10 20:46

## 올바른 실험 (SAE Features)

### 실행 파일
- `extract_L1_31_SAE_CORRECTED.py` - 메인 extraction 스크립트
- `launch_SAE_correct.sh` - 실행 런처 (conda llama_sae_env 사용)

### 실행 방법
```bash
# 새 실험 시작
./launch_SAE_correct.sh 5  # GPU 5 사용

# 또는 tmux에서
tmux new-session -s sae_exp1 "./launch_SAE_correct.sh 5"

# 진행 상황 확인
tmux attach -t sae_exp1
# 나가기: Ctrl+B, D

# 로그 확인
tail -f logs/sae_extraction_*.log
```

### 출력
- **체크포인트**: `/data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_SAE_checkpoint_batch*.json`
- **최종 결과**: `/data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_SAE_features_FINAL_*.json`

## 핵심 수정 사항

### ❌ 기존 방법 (WRONG)
```python
# Raw hidden states 추출 (4096 차원)
layer_hidden = hidden_states[layer][0, -1, :].cpu().numpy()
features[f'layer_{layer}'] = layer_hidden
```

### ✅ 올바른 방법 (CORRECT)
```python
# SAE features 추출 (32,768 차원)
layer_hidden = hidden_states[layer][0, -1:, :]
sae = load_sae(layer)
sae_features = sae.encode(layer_hidden.float())  # SAE 인코딩!
features[f'layer_{layer}'] = sae_features[0].cpu().numpy()
```

## 결과

### 현재 진행 상황
- ✅ Batch 1 (L1-L10): 3,202 significant features
- ✅ Batch 2 (L11-L20): 7,076 significant features  
- 🔄 Batch 3 (L21-L31): 진행 중

### 기존 실험과의 비교

| 구분 | 기존 (WRONG) | 새 실험 (CORRECT) |
|------|-------------|------------------|
| Feature space | Raw hidden states | **SAE features** |
| 차원 | 4,096 per layer | **32,768 per layer** |
| Exp2 호환성 | ❌ Mismatch | ✅ **일치** |

## Archive

참고용 이전 코드는 `archive/` 폴더에 보관:
- `archive/extract_L1_31_features.py` - 원본 raw hidden states 추출 코드
- `archive/exp1.log` - 이전 실험 로그

---
**마지막 업데이트**: 2025-11-10
**작성자**: Claude Code
