# mini GPT 구현 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | SW_AI LAB 302반 |
| 팀명 | 6팀 |
| 팀원 | 송채강, 이규정, 이원재 |

---

## 1. 구현 현황

| 단계 | 구현 내용 | 구현 파일 | 담당자 |
| --- | --- | --- | --- |
| 1 | UTF-8 byte-level BPE tokenizer | `src/bpe.py` |  |
| 2 | GPTDataset, create_dataloader, InputEmbedding | `src/dataset.py`, `src/embeddings.py` |  |
| 3 | MultiHeadAttention, causal mask | `src/attention.py` |  |
| 4 | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate_text_simple | `src/model.py` |  |
| 5 | loss 계산, checkpoint, generate, train_model | `src/train.py` |  |
| 6 | NSMC 감성 분류 Dataset과 classifier | `src/finetune.py` |  |

---

## 2. 테스트 통과 현황

| 실행 명령 | 결과 | 비고 |
| --- | --- | --- |
| `pytest tests/test_bpe.py -v` | 통과 | 반복 corpus에서 가장 자주 등장하는 pair가 merges[0]에 들어가는지 확인 추가 |
| `pytest tests/test_dataset.py -v` | 통과 |  |
| `pytest tests/test_attention.py -v` | 통과 | attention weight가 올바르게 정규화되는지는 확인 테스트 추가 |
| `pytest tests/test_model.py -v` | 통과 |  |
| `pytest tests/test_train.py -v` | 통과 |  |
| `pytest tests/test_finetune.py -v` | 통과 | 짧은 입력은 padding되고, 긴 입력은 truncation되는지 확인하는 테스트 추가 |
| `pytest tests/ -v` | 통과 |  |

---

## 3. 데이터

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NSMC |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전 학습 데이터 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| 미세 조정 데이터 | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식 | 빈 리뷰 제거, 공백 정리, train/validation 분리 |
| 사용한 데이터 크기 | 사전 학습: train 1,379,486자, val 120,560자 |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 인코딩 방식 | 문장 단위 `<bos>`/`<eos>` 추가 |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| vocab_size | 3000 |
| 학습 corpus 크기 | corpus[:1,500,000] / 실제 사용 1,379,486자 |
| 어휘 학습 시간 | 17.01분 (1020.73초) |
| vocabulary 저장 경로 | data\nsmc_bpe_vocab_3000_timed.json |
| 인코딩/디코딩 복원 예시 | decode(encode("이 영화는 좋았다")) == "이 영화는 좋았다" → True |

---

## 5. 모델 구조

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/model.py` |
| 전체 구조 | InputEmbedding -> 4 x TransformerBlock -> LayerNorm -> LM head |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 256 |
| n_heads | 4 |
| n_layers | 4 |
| drop_rate | 0.15 |
| qkv_bias | False |
| stride | 64 |
| learning_rate | 2e-4 |
| weight_decay | 0.03 |
| eval_freq | 200 |
| eval_iter | 10 |
| 총 파라미터 수 | 4,725,248 |

---

## 6. 사전 학습

### 6.1.1 1차 시도

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 구현 | 구현 파일 | `src/train.py` |
| 모델 | vocab_size | 3000 |
| 모델 | context_length | 128 |
| 모델 | stride | 64 |
| 모델 | emb_dim | 256 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 4 |
| 모델 | drop_rate | 0.15 |
| 모델 | qkv_bias | False |
| 학습 | batch_size | 4 |
| 학습 | num_epochs | 100 |
| 학습 | eval_freq | 100 |
| 학습 | eval_iter | 20 |
| 최적화 | learning_rate | 3e-4 |
| 최적화 | weight_decay | 0.01 |

### 6.1.2 결과

| 항목 | 내용 |
| --- | --- |
| final train loss |  |
| final validation loss |  |
| best validation loss |  |
| best epoch |  |
| checkpoint 경로 |  |

#### 손실 그래프

![사전학습 손실 그래프](results/pretrain_loss_curve.png)

- x축: epoch
- y축: train loss, validation loss
- 그래프 아래에 loss 변화 경향을 1~2문장으로 요약

#### 정확도 그래프

![사전학습 정확도 그래프](results/pretrain_accuracy_curve.png)

- x축: epoch
- y축: train accuracy, validation accuracy
- 그래프 아래에 accuracy 변화 경향을 1~2문장으로 요약

#### 생성 샘플

| epoch | 생성 결과 |
| --- | --- |
| 1 |  |
| 2 |  |
| 3 |  |

#### 결과 해석

- validation loss가 가장 낮았던 epoch와 그 이후 추세를 간단히 정리
- 생성 결과가 학습 초반 대비 얼마나 자연스러워졌는지 한두 문장으로 설명

---

## 7. 미세 조정

### 7.1.1 설정

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 구현 | 구현 파일 | `src/finetune.py` |
| 구현 | 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 | 데이터 포맷 | JSONL, `text`, `label` |
| 모델 | tokenizer_vocab_size | 3000 |
| 모델 | max_length | 128 |
| 모델 | emb_dim | 256 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 4 |
| 모델 | drop_rate | 0.2 |
| 모델 | qkv_bias | False |
| 분류기 | classifier_drop_rate | 0.2 |
| 학습 | batch_size | 128 |
| 학습 | num_epochs | 8 |
| 최적화 | learning_rate | 1e-5 |
| 최적화 | weight_decay | 0.02 |

### 7.1.2 결과

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | (예: 128) |
| batch_size | (예: 16) |
| backbone learning rate |  |
| classifier learning rate |  |
| validation loss / accuracy |  |
| test loss / accuracy |  |
| 오류 예시 | 틀린 리뷰 예시와 추정 원인 |

#### 손실 그래프

![미세조정 손실 그래프](results/finetune_loss_curve.png)

- x축: epoch
- y축: train loss, validation loss
- 그래프 아래에 loss 변화 경향을 1~2문장으로 요약

#### 정확도 그래프

![미세조정 정확도 그래프](results/finetune_accuracy_curve.png)

- x축: epoch
- y축: train accuracy, validation accuracy
- 그래프 아래에 accuracy 변화 경향을 1~2문장으로 요약

#### 오류 예시

| 리뷰 문장 | 정답 레이블 | 예측 레이블 | 추정 원인 |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

#### 결과 해석

- best validation loss를 기준으로 어떤 epoch의 모델을 최종 기준으로 삼았는지 정리
- test accuracy가 validation accuracy와 비교해 어떤 흐름을 보였는지 한두 문장으로 설명
- 오분류 예시에서 드러난 한계나 데이터 특성을 간단히 분석

---

## 8. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | (예: Python 3.11) |
| PyTorch | (예: PyTorch 2.x) |
| 실행 환경 | Colab GPU / Colab CPU / 로컬 |
| GPU/CPU 정보 |  |
| 총 학습 소요 시간 |  |

---

## 9. 고찰

- 어려웠던 점
- 한국어 byte-level BPE 구현에서 조심한 점
- loss가 줄어든 이유 또는 줄어들지 않은 이유
- 과적합·과소적합 여부
- 하이퍼파라미터 변경 시도와 결과
- 다음에 개선하고 싶은 점
