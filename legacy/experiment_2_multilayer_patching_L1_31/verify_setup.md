# 실험 2 파일 구조 및 오류 수정 확인

## 1. 이전 오류 (✅ 수정 완료)
- **문제**: All-token patching (모든 토큰 위치를 패치)
- **수정**: Last-token only patching (마지막 토큰만 패치)
- **코드 위치**: experiment_2_L1_31_top300.py, 라인 208-223

## 2. 파일 저장 구조

### 체크포인트 (완전히 분리됨)
- 위치: `/data/llm_addiction/experiment_2_multilayer_patching/`
- 형식: `checkpoint_L{start}_{end}_{process_id}_{timestamp}.json`
- 예시: `checkpoint_L1_2_gpu4_L1_2_20251014_150530.json`
- 저장 주기: 50 features마다

### Response Logs (완전히 분리됨)
- 위치: `/data/llm_addiction/experiment_2_multilayer_patching/response_logs/`
- 형식: `responses_L{start}_{end}_{process_id}_{timestamp}.json`

### 진행 로그 (폴더로 분리)
- 현재: `logs/exp2_gpu{N}_L{start}_{end}.log`
- 아카이브: `logs_archive_20251014_000000/exp2_gpu{N}_L{start}_{end}.log`

## 3. 현재 실행 중인 실험 (16 processes - 안정적 설정)

GPU 4 (4 processes): L1-2, L3-4, L5-6, L7-8
GPU 5 (4 processes): L9-10, L11-12, L13-14, L15-16
GPU 6 (4 processes): L17-18, L19-20, L21-22, L23-24
GPU 7 (4 processes): L25-26, L27-28, L29-30, L31

**GPU 메모리 사용률**: 63.6GB / 81.9GB (77.6%) - 안정적 ✅
**여유 메모리**: 18.3GB per GPU (SAE 로드에 충분)
**실행 시각**: 2025-10-14 16:00 (재시작)

**설정 히스토리**:
- 1 process/GPU (50% 활용, 40GB/80GB) - 메모리 낭비
- 3 processes/GPU (62% 활용, 49.6GB/80GB) - 메모리 낭비
- 4 processes/GPU (78% 활용, 63.6GB/80GB) - ✅ **최적 (현재)**
- 5 processes/GPU (97% 활용, 79.5GB/80GB) - ❌ **OOM 오류** (SAE 로드 불가)

## 4. 검증 완료
- ✅ 수정된 코드 확인됨 (last-token only)
- ✅ 체크포인트 파일명 고유성 확인됨 (timestamp 포함)
- ✅ Response log 파일명 고유성 확인됨 (timestamp 포함)
- ✅ 진행 로그 폴더 분리 확인됨

## 5. 코드 비교 검증 (experiment_2_L1_31_top300.py vs experiment_2_final_correct.py)

### Prompts (✅ 완전히 동일)
- **risky_prompt**: 동일한 게임 히스토리, 잔액, 선택지
- **safe_prompt**: 동일한 게임 히스토리, 잔액, 선택지

### Patching Method (✅ 핵심 로직 동일)
- **Last-token 추출**: `last_token = hidden[:, -1:, :].float()` (동일)
- **SAE 인코딩**: `features = sae.encode(last_token)` (동일)
- **Feature 패칭**: `features[0, 0, feature_id] = float(patch_value)` (동일)
- **SAE 디코딩**: `sae.decode(features)` (동일)
- **히든 교체**: `hidden[:, -1:, :] = reconstructed` (동일)

**차이점**: Hook 타입만 다름 (pre-hook vs forward-hook), 하지만 패칭 동작은 완전히 동일
