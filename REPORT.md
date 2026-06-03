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
| 모델 | emb_dim | 128 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 4 |
| 모델 | drop_rate | 0.1 |
| 모델 | qkv_bias | False |
| 학습 | batch_size | 16 |
| 학습 | num_epochs | 30 |
| 학습 | eval_freq | 50 |
| 학습 | eval_iter | 20 |
| 최적화 | learning_rate | 3e-4 |
| 최적화 | weight_decay | 0.02 |

### 6.1.2 epoch 별 loss

| epoch | step | train_loss | val_loss | train_acc | val_acc | val_ppl |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 650 | 7.7274 | 7.7192 | 0.0085 | 0.0088 | 2251.13 |
| 2 | 1350 | 7.3395 | 7.3895 | 0.0168 | 0.0147 | 1618.90 |
| 3 | 2050 | 6.6247 | 6.7735 | 0.0429 | 0.0372 | 874.33 |
| 4 | 2750 | 6.2123 | 6.4438 | 0.0636 | 0.0543 | 628.82 |
| 5 | 3450 | 5.9857 | 6.2787 | 0.0758 | 0.0644 | 533.08 |
| 6 | 4150 | 5.8006 | 6.2019 | 0.0863 | 0.0673 | 493.71 |
| 7 | 4850 | 5.6427 | 6.1555 | 0.0952 | 0.0692 | 471.31 |
| 8 | 5550 | 5.5222 | 6.1328 | 0.0973 | 0.0667 | 460.74 |
| 9 | 6250 | 5.4302 | 6.1347 | 0.1068 | 0.0663 | 461.61 |
| 10 | 6950 | 5.3255 | 6.1415 | 0.1110 | 0.0677 | 464.73 |
| 11 | 7650 | 5.2403 | 6.1628 | 0.1183 | 0.0648 | 474.74 |
| 12 | 8350 | 5.1734 | 6.1891 | 0.1229 | 0.0640 | 487.40 |
| 13 | 9000 | 5.0728 | 6.2187 | 0.1322 | 0.0619 | 502.03 |
| 14 | 9700 | 5.0143 | 6.2636 | 0.1379 | 0.0639 | 525.09 |
| 15 | 10400 | 4.8923 | 6.3101 | 0.1462 | 0.0623 | 550.10 |
| 16 | 11100 | 4.8206 | 6.3539 | 0.1541 | 0.0619 | 574.73 |
| 17 | 11800 | 4.7704 | 6.4083 | 0.1638 | 0.0593 | 606.88 |
| 18 | 12500 | 4.6695 | 6.4511 | 0.1730 | 0.0608 | 633.37 |
| 19 | 13200 | 4.6428 | 6.5082 | 0.1753 | 0.0578 | 670.59 |
| 20 | 13900 | 4.5828 | 6.5500 | 0.1825 | 0.0584 | 699.25 |
| 21 | 14600 | 4.5494 | 6.5966 | 0.1869 | 0.0577 | 732.59 |
| 22 | 15300 | 4.4570 | 6.6496 | 0.2013 | 0.0565 | 772.46 |
| 23 | 16000 | 4.4056 | 6.6932 | 0.2089 | 0.0539 | 806.92 |
| 24 | 16700 | 4.3515 | 6.7386 | 0.2135 | 0.0542 | 844.42 |
| 25 | 17400 | 4.3093 | 6.7753 | 0.2204 | 0.0554 | 875.92 |
| 26 | 18050 | 4.2904 | 6.8208 | 0.2218 | 0.0518 | 916.73 |
| 27 | 18750 | 4.2319 | 6.8889 | 0.2292 | 0.0526 | 981.28 |
| 28 | 19450 | 4.1824 | 6.9509 | 0.2346 | 0.0517 | 1044.13 |
| 29 | 20150 | 4.1325 | 7.0021 | 0.2461 | 0.0491 | 1098.96 |
| 30 | 20850 | 4.0704 | 7.0678 | 0.2504 | 0.0500 | 1173.53 |

### 6.1.3 결과

| 항목 | 내용 |
| --- | --- |
| final train loss | 4.0704 |
| final validation loss | 7.0678 |
| best validation loss | 6.1328 |
| best epoch | 8 |

#### 손실 그래프


<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/68e21829-9c81-440c-b06f-fe004a67968c" />

- 그래프를 보면 train_loss는 epoch가 증가할수록 지속적으로 감소한다. 이는 모델이 학습 데이터에 대해서는 점점 더 잘 맞춰지고 있음을 의미한다. 반면 val_loss는 초반에는 빠르게 감소하지만, 약 8~10 epoch 부근에서 가장 낮은 값을 기록한 뒤 다시 증가하는 모습을 보인다.

#### 정확도 그래프

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/a6d9e711-d1e4-42bf-b9d1-99200f4a1391" />


#### 생성 샘플

| epoch | 생성 샘플 |
|---:|---|
| 1 | 영화고이기이은 영화가.을을도지..하고만의고한이도. .도과가가도만가은,을..로한도이..가 을로이영화에...은아과.가이가은을의는아..고에 에 영화다..이다 가하고만지가가은를. |
| 2 | 영화다 !! 이 4리. 그은 시간의 좋았어서 우이 된음 에 하고 지루을 하다. 나를 더 좋다. 한국을 일도 어지하수은 정말 어�가...... 3에 한기...? 오로 진짜 있다. 이런 4 어가는 이이 이를 볼, 보긴다.. 한 시도 보고 말을 왜 |
| 3 | 영화! 이하는 사람에 영화~!!?? fegueh Alidasestobciliasteittiethot hevestiesteelos ht etespd wlnot fhee |
| 4 | 영화인데. 그래도 저리지만 더불어 이뻐 마지막, 김상하는 거짓을 남는 건지훈을 쏭제에서 봤는데 이런걸만하는 걸리기싫어졌을까 제, ㄱㄱ♥♥!! 우시카나, 어쨌는 장면의영화만도 아까운 영화 명작.444.35년전작. "볼수있 |
| 5 | 영화임새가있다. 영화보던거지로 빌려온다. 아들이 좀 더 재밌지만... 내..이건 뭐다 이거보고있다 그것에 보지면서 봤어야 했다 그리다살다 개발만하는 거 같아직도 못해가능한장면도 많이 준다 뭐니 기대없이 웃게되는 영화! 너무 재밌어놨는데 정말 재밌는데 |

#### 결과 해석

- 첫 번째 실험에서는 epoch를 30으로 설정하고, dropout_ratio를 0.1, weight_decay를 0.02로 설정하여 기본적인 학습 경향과 과적합 여부를 확인하였다. 실험 결과, 초반에는 train_loss와 val_loss가 모두 감소하며 정상적으로 학습이 진행되었지만, 중반 이후에는 train_loss만 계속 감소하고 val_loss는 다시 증가하였다.
- 그래프를 보면 train_loss는 epoch가 증가할수록 지속적으로 감소한다. 이는 모델이 학습 데이터를 점점 더 잘 학습하고 있음을 의미한다. 반면 val_loss는 초반에는 train_loss와 함께 빠르게 감소하지만, 약 8~10 epoch 부근에서 가장 낮은 값을 기록한 뒤 다시 증가하는 모습을 보인다.
- 이는 모델이 일정 시점 이후부터 학습 데이터에는 계속 적응하지만, 검증 데이터에 대한 일반화 성능은 더 이상 개선되지 않는다는 것을 의미한다.

#### 가설과 기대 결과
- 첫 번째 테스트에서는 train_loss가 지속적으로 감소하는 반면 val_loss는 약 8 epoch 이후 다시 증가하였다.
- 이는 모델이 학습 데이터에는 계속 적응하지만 validation 데이터에 대한 일반화 성능은 악화되는 과적합 현상으로 해석된다.
- 따라서 두 번째 테스트에서는 drop_rate와 weight_decay를 증가시키고 learning_rate를 낮춰 정규화를 강화하고 업데이트 폭을 줄인다.
- 이를 통해 train_loss는 계속 감소하되, val_loss의 상승 시점을 늦추고 상승 폭을 완화할 수 있을 것으로 기대한다.
- 
### 6.2.1 2차 시도

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 모델 | **drop_rate** | **0.2** |
| 학습 | num_epochs | 20 |
| 최적화 | learning_rate | 2e-4 |
| **최적화** | **weight_decay** | **0.03** |

### 6.2.2 결과

| 항목 | 내용 |
| --- | --- |
| final train loss | 5.0059 |
| final validation loss | 6.2184 |
| best validation loss | 6.0851 |
| best epoch | 11 |

#### 손실 그래프

<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/bc20924b-9280-49d6-bd9c-28bf64018d82" />

#### 정확도 그래프

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/8d8e3d8e-5c22-4ca8-9684-f2ba2f3e4dcc" />

#### 결과 해석

- 두 번째 테스트에서도 train_loss는 지속적으로 감소하므로 학습 데이터에 대한 학습은 정상적으로 진행되었다.
- val_loss는 초반에 빠르게 감소한 뒤 약 10~13 epoch 부근에서 가장 낮은 구간을 형성한다.
- 이후 val_loss가 급격히 증가하지는 않지만, 후반부에서 여전히 약간씩 증가하는 모습을 보인다.
- 또한 learning rate를 낮추고 dropout 및 weight decay를 증가시켰기 때문에, 같은 epoch 수에서 train loss는 첫 번째 테스트보다 덜 감소하였다.
- 실제로 validation loss는 첫 번째 테스트보다 더 오랫동안 낮은 구간을 유지했고, 후반부 상승 폭도 완만해져 과적합이 일부 완화되었다.
- 따라서 첫 번째 테스트보다 과적합이 비교적 완화되었지만, train_loss와 val_loss의 차이가 계속 벌어지는 점에서 과적합 경향이 완전히 사라진 것은 아니다.

### 6.3.1 3차 시도

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 구현 | 구현 파일 | `src/train.py` |
| 실험 목적 | 변경 포인트 | 하이퍼파라미터 조정보다 인코딩 방식 변경의 효과 확인 |
| 데이터 | 샘플 단위 | 리뷰 한 줄을 하나의 독립적인 문장으로 처리 |
| 인코딩 | 변경 전 | 전체 학습 텍스트를 한 번에 `tokenizer.encode(train_text)` |
| 인코딩 | 변경 후 | 각 리뷰 라인을 개별적으로 encode하고 `<bos>`, `<eos>` 추가 |
| 인코딩 | 기대 효과 | 리뷰 경계를 명시하여 문장 간 비연속 전환 학습 완화 |

#### 인코딩 방식 비교

변경 전:

```python
tokenizer.encode(train_text)
```

변경 후:

```python
for line in train_text.splitlines():
    token_ids.extend(tokenizer.encode(line, add_bos_eos=True))
```

### 6.3.2 결과

| 항목 | 내용 |
| --- | --- |
| final train loss | 4.2629 |
| final validation loss | 5.0643 |
| best validation loss | 5.0265 |
| best epoch | 13 |
| 주요 관찰 | 하이퍼파라미터 조정만으로는 `val_loss`가 의미 있게 감소하지 않았다. |
| 핵심 변경 | 리뷰 단위 encode + `<bos>`/`<eos>` 추가 |
| 결과 해석 | 단순 하이퍼파라미터 조정보다 인코딩 방식 변경이 `val_loss` 개선에 더 직접적인 영향을 주었다. |

#### 손실 그래프

<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/c25c14fd-fd0b-4a85-95b8-f8adcc214171" />

#### 정확도 그래프

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/eb379b14-40e7-4582-b692-61c851e011f0" />

#### 생성 샘플

| epoch | 생성 결과 |
| --- | --- |
| 1 | 영화다 못보고 평점 남주겠어볼껀면 보아봄 제발 욕밖에 맞추려고 했는데.. 이거보단 잼있게 만드는 영화 찍어? 좋은 어째 보는데 나서 좋았는데 중심도 못 주려했는데 너무 재밌네요. 뭐라고요ㅋ... 1,2까지만 정말 재밌고 빤오랫동안 느슨 너무 높을 죽여? 감독의 상상력...저기서 |
| 2 | 영화중에 가장 인상깊지 않은 느낌. 보고싶어지는 것도 그랬어요.. 스토리도 죽지하루크트콤한영화 2-,2탄과 편집도 드는,7년의 달달콤한 소재는, 액션.미,20대와 연출, 모든게울언,명배우를 가야 해보들의 향한 소재는,원작,해. 왜곡도 많아서버려야 |
| 3 | 영화보고 왜 0점 알아라. 근데 왜 그랬보다가 딴작품이다. 여기들' 늦게만 안나옴..그는 터져야?박작지ㅋㅋㅋ여배우가요?나물과 브로써 나발한다는지도하시면 시나리오가? 1점도 아까운 둘째치고 애니가? 정말 잘봣는영화 재밌어가는건 둘,음악의 새로운 시리, 윤성이 있다. 재밌 |

#### 결과 해석

- 하이퍼파라미터를 조정해도 `val_loss`가 이전 실험 대비 의미 있게 줄어들지는 않았다. 즉, 단순히 epoch, dropout, weight decay 같은 학습 설정만 바꾸는 것으로는 검증 성능 개선에 한계가 있었다.
- 원인 중 하나는 기존 학습용 encode 방식에서 각 리뷰의 시작과 끝을 나타내는 특수 토큰 `<bos>`, `<eos>`가 빠져 있었다는 점이다. NSMC 데이터는 한 줄이 하나의 독립적인 리뷰이므로, 전체 텍스트를 하나의 긴 시퀀스로 이어 붙이면 리뷰 간 경계가 모델에 명확히 전달되지 않는다.
- 기존 방식에서는 리뷰 A의 마지막 토큰 다음에 리뷰 B의 첫 토큰이 바로 이어진다. 이 경우 모델은 의미적으로 연결되지 않은 두 리뷰 사이의 전환까지 다음 토큰 예측 대상으로 학습하게 되어, 실제 문장 구조와 다른 패턴까지 함께 학습할 가능성이 있다.
- 이를 개선하기 위해 각 리뷰 라인을 개별적으로 encode하고, 각 라인의 앞뒤에 `<bos>`, `<eos>` 토큰을 추가하는 방식을 적용하였다. 이 방식은 모델에게 각 리뷰의 시작과 끝을 명시적으로 알려 주어, 문장 단위 구조를 더 자연스럽게 학습하도록 돕는다.
- 결과적으로 이번 실험에서는 단순한 하이퍼파라미터 조정보다 encode 방식 변경이 `val_loss` 개선에 더 직접적인 영향을 주었다고 해석할 수 있다.

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
| 최적화 | learning_rate | 2e-5 |
| 최적화 | weight_decay | 0.02 |

### 7.1.2 결과

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | 128 |
| batch_size | 128 |
| backbone learning rate | 2e-5 |
| classifier learning rate | 2e-5 |
| validation loss / accuracy | 0.4169 / 0.8218 |
| test loss / accuracy | 0.4146 / 0.8274 |

|  | **epoch** | **train_loss** | **val_loss** | **train_acc** | **val_acc** |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0.662586 | 0.509556 | 0.639062 | 0.748146 |
| 1 | 2 | 0.549443 | 0.482957 | 0.722108 | 0.770231 |
| 2 | 3 | 0.503963 | 0.462987 | 0.754044 | 0.781398 |
| 3 | 4 | 0.475936 | 0.452796 | 0.771783 | 0.794066 |
| 4 | 5 | 0.456949 | 0.436055 | 0.784001 | 0.805734 |
| 5 | 6 | 0.440778 | 0.445127 | 0.793740 | 0.800150 |
| 6 | 7 | 0.427146 | 0.418713 | 0.801364 | 0.816735 |
| 7 | 8 | 0.415599 | 0.442509 | 0.807103 | 0.816068 |
| 8 | 9 | 0.407880 | 0.433760 | 0.812589 | 0.817318 |
| 9 | 10 | 0.398865 | 0.416875 | 0.817401 | 0.821818 |
| 10 | 11 | 0.389004 | 0.416571 | 0.824009 | 0.823985 |

#### 손실 그래프

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/79addd18-6aba-4b9f-acef-b3aa08edd213" />

- Fine-tuning 결과, train loss와 validation loss가 모두 전반적으로 감소했다.
- validation loss는 일부 epoch에서 작은 흔들림이 있었지만 전체적으로 하락 추세를 유지했다.

#### 정확도 그래프

<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/19ae2319-938f-4324-8206-e87d2489b013" />

- train accuracy와 validation accuracy도 함께 상승하였다.
- validation accuracy도 약 0.825까지 상승하였다.

#### 오류 예시

| text | pred_label | negative_prob | positive_prob |
| --- | --- | ---: | ---: |
| 이 영화는 정말 좋았다 | 긍정 | 0.007649 | 0.992351 |
| 시간이 너무 아까웠다 | 부정 | 0.989832 | 0.010168 |
| 배우 연기는 좋았지만 이야기는 지루했다 | 부정 | 0.906879 | 0.093121 |
| 기대보다 훨씬 재미있었다 | 긍정 | 0.021235 | 0.978765 |
| 다시는 보고 싶지 않은 영화였다 | 긍정 | 0.038633 | 0.961367 |

#### 결과 해석

- best validation loss를 기준으로 validation loss가 가장 낮았던 epoch의 checkpoint를 최종 모델 기준으로 삼았다. 마지막 epoch 모델은 train loss가 더 낮더라도 validation 성능이 정체되거나 악화될 수 있으므로 best checkpoint를 선택하였다.

- test accuracy는 validation accuracy와 큰 차이 없이 비슷한 수준을 보여, validation에서의 성능이 test data에서도 대체로 유지되었다고 볼 수 있다. 이는 모델이 특정 validation에만 과하게 맞춰진 것은 아니라는 뜻이 된다.

- 오분류 예시인 “다시는 보고 싶지 않은 영화였다”에서는 부정 표현 “다시는 ~지 않은”보다 “보고 싶다”라는 긍정 표현에 더 강하게 반응한 것으로 보인다. 이는 짧은 리뷰 안에서도 부정/반전 표현을 정확히 해석하는 데 한계가 있음을 보여준다.

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
