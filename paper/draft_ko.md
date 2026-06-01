# Florence-2 기반 이미지 설명의 시각적 근거 검증 및 Hallucination 탐지

**Visual Evidence Verification for Image Caption Hallucination Detection using Florence-2**

> IPIU2026 워크샵 양식 (`Team/IPIU2026_sample/IPIU2026_sample.doc`) 으로 옮겨담을 한국어 논문 원고. 본문 4~6쪽 분량.
> 양식 규칙 — 제목 바탕체 14 bold, 본문 한글 바탕체 10 / 영문 Times New Roman 10, 장 번호 돋움체 12 bold, 표 캡션 상단·그림 캡션 하단 가운데 정렬, 발표자 이름 위에 `o` 표시.

---

## 요 약

본 연구는 통합 비전-언어 모델 Florence-2[1]의 captioning과 object detection을 결합하여, 같은 모델 내에서 이미지 설명의 시각적 근거를 검증하는 **self-verification 파이프라인**을 제안한다. 파이프라인은 (i) `<CAPTION>` 결과에서 객체 mention을 추출하고, (ii) 동일 이미지의 `<OD>` 결과와 정규화된 라벨로 매칭하여 supported / unsupported / hallucination candidate 세 분류로 판정한다. COCO val2017 500장 (mentions 732건) 에서 Florence-2-large-ft는 baseline hallucination rate를 2.46%에서 검증 후 0.70%로 감소시켰으며 (95% CI 둘 사이 겹침 없음), unsupported 탐지 F1 = 0.812, grounding accuracy@0.5 = 95.65% 를 달성했다. 외부 POPE benchmark[3] 에서는 random / popular / adversarial 세 split 모두에서 LLaVA-1.5-7B, InstructBLIP을 넘는 89.97~91.34% 정확도를 보였다. 또한 동의어 매핑 정책에 대한 ablation을 통해, *과한 매핑* (예: `table → dining table`) 이 baseline hallucination을 ~80% 부풀린다는 것을 정량 입증하고, 정직한 평가에 필요한 strict 어휘 정책을 제시한다. Cross-model 분석 (BLIP caption + Florence-2 OD) 에서는 captioner를 바꾸어도 검증의 recall (0.571) 이 동일하게 유지되어, **검증 성능의 상한이 captioner의 hallucination 양이 아니라 OD 모델의 small-object recall 한계에 의해 결정됨**을 밝힌다. 본 연구는 단일 통합 VLM 내 self-verification의 실용적 효과와 구조적 한계를 함께 보고하는 첫 시도이다.

## 1. 서론

Florence-2[1], BLIP-2, Qwen-VL, LLaVA 등 통합 비전-언어 모델(Vision-Language Model, VLM)이 자연스러운 이미지 설명을 생성하지만, **실제 이미지에 존재하지 않는 객체를 caption에 포함하는 object hallucination 문제**[3,4]가 지속적으로 보고되고 있다. 사용자가 caption만 보고 정보를 받아들이는 응용 — 접근성 보조, 검색 인덱스, 자동 보고서 — 에서 hallucination은 신뢰성을 직접 저해한다.

대표적 평가 방법인 CHAIR[3], POPE[4]는 caption mention과 데이터셋 GT 또는 yes/no 질문 응답을 비교한다. 이는 평가 시점엔 유용하지만 **사용 시점**에는 GT가 없으므로 적용할 수 없다. 한편 grounding/detection 단계에서 별도 모델을 사용하는 검증 방식은 추가 모델 비용을 요구한다.

본 연구는 Florence-2가 captioning과 grounding을 **단일 prompt-conditioned 모델로 동시에 수행**할 수 있다는 점에 주목하여, 별도 모델 없이 자신이 생성한 caption을 자신의 visual grounding 결과로 재검증하는 self-verification 파이프라인을 설계·평가한다. 주된 기여는 다음과 같다.

1. Florence-2 단일 모델 위에서 동작하는 caption-grounding consistency 검증 파이프라인의 설계와 구현.
2. COCO 80 카테고리 어휘에 대한 **strict synonym 매핑** 정책 제안 — 기존의 과한 매핑이 hallucination rate 를 인위적으로 부풀리는 효과를 정량 입증.
3. COCO val2017 500장 본 실험 (732 mentions, 95% bootstrap CI 포함) 에서 baseline 2.46% → verified 0.70% 의 통계적으로 유의한 감소, F1 = 0.812.
4. **외부 POPE benchmark**에서 89.97~91.34% 정확도로 LLaVA, InstructBLIP 등 대형 LVLM 상회.
5. Cross-model 분석을 통한 **검증 한계의 원인 분리**: 검증 recall이 captioner와 무관 → 한계는 OD 모델의 small-object recall에서 발생.

## 2. 관련 연구

**Florence-2[1]**는 captioning, object detection, phrase grounding, segmentation, OCR 등 다양한 비전·비전-언어 과제를 *task prompt* 만 바꿔 단일 sequence-to-sequence 모델로 수행한다. FLD-5B 라는 대규모 multi-task 어노테이션으로 학습되어 작은 모델 크기에도 강한 성능을 보인다. 본 연구는 `<CAPTION>`, `<DETAILED_CAPTION>`, `<OD>`, `<CAPTION_TO_PHRASE_GROUNDING>` prompt 를 활용한다.

**Object hallucination 평가**. Rohrbach 등[3]은 CHAIR(Caption Hallucination Assessment with Image Relevance)를 도입해 caption mention이 GT 객체 집합에 포함되는지를 평가한다. POPE[4]는 "Is there a X in the image?" yes/no polling 으로 hallucination을 더 큰 LVLM 환경에서 비교한다. 본 연구는 두 metric의 변형을 모두 사용한다.

**COCO[2]**. 80개 객체 카테고리와 이미지당 캡션·인스턴스 어노테이션을 제공. hallucination 평가의 사실상 표준.

## 3. 방법

### 3.1 전체 파이프라인

본 시스템은 4단계 단방향 파이프라인이다 (그림 1).

1. **Caption 생성**: Florence-2 `<CAPTION>` 또는 `<DETAILED_CAPTION>` prompt 로 입력 이미지에 대한 caption을 생성한다.
2. **Object mention 추출**: COCO 80 카테고리 어휘 + **strict synonym 매핑 테이블**을 사용해 caption에서 객체 mention을 추출한다.
3. **시각적 검증**: 같은 이미지에 `<OD>` prompt 를 한 번 호출, 검출된 모든 (라벨, bbox) 쌍을 동일 매핑으로 canonical 라벨로 정규화 후 각 mention의 canonical과 매칭한다.
4. **분류 및 시각화**: mention을 supported / unsupported로 분류한다. unsupported AND GT에도 부재하면 **hallucination candidate**로 표시한다.

### 3.2 Strict synonym 매핑 정책

caption의 자유 어휘를 COCO80 정식 라벨로 정규화하기 위해 동의어 매핑 테이블을 사용한다. 본 연구는 다음 두 변형을 비교한다.

- **aggressive (legacy)**: `table → dining table`, `baseball → sports ball`, `monitor → tv`, `stove → oven`, `plant → potted plant`, generic `glove → baseball glove` 등 *generic-to-specific* 매핑을 포함.
- **strict (ours)**: 위 generic-to-specific 매핑을 제거. 단순 복수형(`bikes → bicycle`)·언어 변형(`television → tv`, `cellphone → cell phone`)·의미 동치(`sofa → couch`)만 유지.

3.3절 ablation에서 보이듯, aggressive vocab은 baseline hallucination rate를 약 80% 부풀리며, 본 연구의 모든 주 실험은 **strict vocab**을 기본값으로 한다.

### 3.3 검증 (Verification) 전략

`<OD>` 결과의 각 라벨을 strict 매핑으로 canonical 라벨로 정규화하고, mention의 canonical과 일치하면 supported, 첫 매칭의 bbox가 시각적 근거로 부여된다. 대안으로 mention 1개당 `<CAPTION_TO_PHRASE_GROUNDING>` 을 호출하는 방식도 구현했으나, 호출 횟수가 mention 수에 비례해 비용이 크고 `<OD>` 라벨 매칭과 결과 차이가 본 실험 규모에서 미미하여 본 보고서의 주된 결과는 `<OD>` 전략을 채택했다.

### 3.4 평가 지표

| 지표 | 정의 |
| --- | --- |
| Hallucination Rate | (canonical ∉ image GT) / total mentions |
| Verified Hallucination Rate | (canonical ∉ GT) / supported mentions |
| Unsupported P / R / F1 | TP=(예측 unsupp AND ∉GT), 등 |
| Grounding Acc@0.5 | supported bbox 와 동일 카테고리 GT bbox 의 max IoU ≥ 0.5 비율 |
| POPE accuracy / F1 | yes/no 응답의 4-cell confusion |

본 연구의 모든 비율 지표는 percentile bootstrap (n=1000, seed=42) 95% 신뢰구간을 함께 보고한다.

## 4. 실험

### 4.1 데이터셋
- **주 실험**: COCO val2017 (5,000장) 에서 시드 7로 무작위 추출한 500장 (`instances_val2017.json` 전체 ID 풀에서 샘플링).
- **POPE**: 표준 POPE 평가 — random / popular / adversarial 세 split, 각 3,000 yes/no 질문, 500개 val2014 unique images.
- **Cross-model**: 주 실험 500장의 앞쪽 200장.

### 4.2 모델 / 환경
- 주 모델: `microsoft/Florence-2-large-ft` (~770M parameters).
- Cross-model: `Salesforce/blip-image-captioning-large` (~470M parameters).
- 보조 분석: `microsoft/Florence-2-base-ft` (~230M).
- 하드웨어: NVIDIA RTX 3070 (8GB VRAM).
- 라이브러리: PyTorch 2.5.1 + CUDA 12.1, `transformers==4.49.0` (Florence-2 trust_remote_code 호환), fp16 추론.
- 디코딩: `num_beams=3`, `do_sample=False`. 추가 학습 없는 zero-shot inference.

### 4.3 Baseline / 제안 방법
- **Baseline**: `<CAPTION>` 결과의 모든 mention을 그대로 인정. Hallucination rate = (GT에 없는 mention) / total.
- **Proposed**: `<OD>` 로 검증해 supported mention만 인정. unsupported를 hallucination candidate로 표시·제거.

## 5. 결과

### 5.1 주 결과 — 500장 strict CAPTION

**표 1. Florence-2-large-ft, COCO val2017 500장, strict vocab, `<CAPTION>` prompt 주 실험 결과** (95% bootstrap CI)

| Method | Mentions | Baseline Hall% | Verified Hall% | F1 (Unsupp) | Grd Acc@0.5 |
| --- | --: | --: | --: | --: | --: |
| `<CAPTION>` baseline | 732 | 2.46 [1.50, 3.55] | – | – | – |
| **Ours (verified)** | **718** | – | **0.70 [0.14, 1.39]** | **0.812 [0.636, 0.941]** | **95.65** |

검증을 통해 hallucination rate가 통계적으로 유의하게 감소 (CI 겹침 없음). Confusion matrix: TP=13, FP=1, FN=5, TN=713 → Precision=0.929, Recall=0.722.

### 5.2 `<DETAILED_CAPTION>` 보조 실험

**표 2. 같은 500장에 대해 `<DETAILED_CAPTION>` 으로 실험**

| Method | Mentions | Baseline Hall% | Verified Hall% | F1 | Grd Acc@0.5 |
| --- | --: | --: | --: | --: | --: |
| `<DETAILED_CAPTION>` baseline | 1121 | 6.07 [4.73, 7.40] | – | – | – |
| Ours (verified) | 1071 | – | **2.99 [2.05, 4.01]** | 0.610 [0.495, 0.710] | 93.94 |

DETAILED는 mention 53% 더 많지만 baseline rate가 **2.5× 더 높음** (긴 caption은 더 hallucinate, 문헌과 일치[4]). FP 14건은 대부분 작은 식기류 (bowl, spoon, bottle) 와 욕실 객체 (sink) — *captioner는 정확하지만 OD가 작은 객체를 못 잡음*.

### 5.3 POPE 외부 평가

**표 3. POPE benchmark — Florence-2-large-ft (strict vocab)**

| Split | Acc | Precision | Recall | F1 |
| --- | --: | --: | --: | --: |
| random | **91.34** | **99.27** | 83.65 | **90.79** |
| popular | **90.83** | 97.36 | 83.65 | **89.99** |
| adversarial | **89.97** | 95.53 | 83.65 | **89.20** |

문헌의 reported 수치와 비교 (random / popular / adversarial):
- MiniGPT-4: 78.95 / 75.18 / 72.05
- mPLUG-Owl2: 86.06 / 84.50 / 83.20
- InstructBLIP: 87.96 / 86.10 / 83.10
- LLaVA-1.5-7B: 88.07 / 87.20 / 84.27
- **Florence-2-large-ft (본 연구)**: **91.34 / 90.83 / 89.97**

Florence-2 OD가 POPE 3 split 모두에서 위 LVLM들을 능가한다. Precision이 매우 높음 (95~99%) — *OD는 보수적이라 false positive를 거의 만들지 않음*. Recall이 83.65%로 일관되게 나타나는데, 이는 *OD가 16% 정도의 객체를 못 잡는 구조적 한계* 이다.

### 5.4 Cross-model 분석

**표 4. 같은 200장에서 self vs cross 검증** (95% CI 포함 본문 생략)

| Setting | Mentions | Baseline | Verified | P | R | F1 | TP |
| --- | --: | --: | --: | --: | --: | --: | --: |
| **Self** (F2 caption + F2 OD) | 290 | 2.41 | 1.05 | 1.00 | 0.571 | 0.727 | 4 |
| **Cross** (BLIP caption + F2 OD) | 298 | **4.70** | 2.07 | 1.00 | **0.571** | 0.727 | **8** |

BLIP-large가 Florence-2 보다 약 2× 자주 hallucinate (baseline 4.70 vs 2.41) 하지만, **검증의 recall은 0.571로 두 조건에서 정확히 동일**. → 검증 시스템의 한계는 captioner 가 무엇이냐가 아니라 *OD가 객체를 일정 비율로 못 잡는다* 는 점에서 발생함. 같은 결론이 POPE recall 83.65%와 일치.

### 5.5 Synonym vocab Ablation

**표 5. 같은 500장 CAPTION 에서 vocab 정책만 변경**

| Vocab | Mentions | Baseline | Verified | F1 |
| --- | --: | --: | --: | --: |
| **strict (ours)** | 732 | **2.46** | **0.70** | 0.812 |
| aggressive (legacy) | 782 | 4.48 | 1.46 | 0.787 |

aggressive 매핑은 baseline rate를 **~80% 부풀리지만** (2.46 → 4.48) F1은 거의 변화 없음. 추가 mention 50건의 대다수가 `table`, `baseball`, `monitor`, `plant`, `stove`, `glove` 같은 generic-to-specific 매핑에서 만들어진 "거짓 hallucination" 이며 이는 *어휘 정책의 artifact* 이다. **이전 문헌의 자가검증 효과 일부는 strict 정책으로 재평가하면 사라진다**.

### 5.6 정성 분석 (그림 2~4)

- **그림 2 — Joint failure**: image_id 5503 *"A person standing in front of a toilet"* — GT에는 `toilet`만, 사람 없음. Florence-2 caption과 OD 둘 다 `person`을 잘못 인식 → 검증 통과. 동일 모델 self-verification의 본질적 사각.
- **그림 3 — OD recall 한계**: image_id 64084 detailed caption *"plates, spoons, forks, bottles..."* — GT에 spoon 존재. caption은 정확. 그러나 OD가 작은 spoon을 못 잡음 → FP (over-flag).
- **그림 4 — 검증 성공**: image_id 78426 caption *"...a phone..."* — GT에 cell phone 없음, OD에도 없음 → 정확히 unsupported로 깃발.

## 6. 논의

### 6.1 핵심 발견 요약

1. **strict vocab + 500장 본 실험**에서 검증이 hallucination을 통계적으로 유의하게 감소 (2.46 → 0.70%, CI 겹침 없음, F1 0.812).
2. **POPE 외부 평가**에서 Florence-2 OD가 LLaVA·InstructBLIP·mPLUG-Owl2 등을 모두 상회 (90~91% acc).
3. **검증 recall의 상한은 OD의 small-object detection 한계** — captioner 변경(cross-model)도 같은 0.571 recall, POPE 도 83.65% recall.
4. **이전 보고된 self-verification 성능 향상의 일부는 vocab 매핑 artifact** — strict 정책으로 재평가 시 baseline이 절반으로 감소.

### 6.2 한계
1. **본질적 self-verification 사각**: caption과 OD 둘 다 사람·dining table·앵벌이를 동시 잘못 인식하는 joint failure는 잡지 못함. 정성 그림 2가 직접 증거.
2. **OD recall 한계** (~16% 누락) — small/occluded/partial 객체. POPE 와 본 실험 모두 동일한 한계 표출.
3. **어휘 한정**: COCO80 밖 객체(tripod, plate, rice 등) 는 추출 자체 불가능.
4. **Caption correction 단순함**: verified caption은 unsupported mention을 *제거*만. 보다 부드러운 보정(신뢰도 표시, 부분 수정) 은 향후.
5. **본 실험은 yes/no 또는 객체 존재 hallucination에 한정**. attribute/relation hallucination은 측정 범위 밖.

### 6.3 향후 방향
1. captioner와 verifier를 분리해 *서로 다른 모델*로 운용 (본 연구 Cross-model 실험이 첫 step).
2. SAM2 등 segmentation 결합으로 더 정밀한 visual evidence.
3. 더 큰 OD 모델 또는 open-vocabulary detector (GroundingDINO, OWL-ViT) 로 small-object recall 개선.
4. attribute / relation hallucination 까지 확장.
5. 한국어 caption verification.

## 7. 결론

본 연구는 Florence-2의 통합 multi-task 능력을 활용해 caption-grounding consistency를 검증하는 self-verification 파이프라인을 제안·평가했다. COCO val2017 500장 strict vocab 본 실험에서 hallucination rate가 2.46%에서 0.70%로 통계적으로 유의하게 감소 (F1 = 0.812), POPE 3 split 모두에서 기존 LVLM들을 상회 (89.97~91.34%) 했다. 동시에 ablation을 통해 *과한 synonym 매핑이 hallucination 측정을 부풀리는 artifact* 임을 정량 입증하고, cross-model 비교로 *검증의 recall 한계가 OD 모델 자체의 구조적 결손에 의한 것임* 을 분리해 보였다. 본 연구는 단일 통합 VLM 만으로 실용적 hallucination 검출이 가능함을 보임과 동시에, 그 한계의 정확한 원인을 처음으로 분리 보고한다.

## 감사의 글

본 연구는 컴퓨터비전 과목 프로젝트의 일환으로 수행되었다.

## 참고문헌

[1] B. Xiao et al., "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks," CVPR 2024.

[2] T. Y. Lin et al., "Microsoft COCO: Common Objects in Context," ECCV 2014.

[3] A. Rohrbach et al., "Object Hallucination in Image Captioning," EMNLP 2018.

[4] Y. Li et al., "Evaluating Object Hallucination in Large Vision-Language Models (POPE)," EMNLP 2023.

[5] J. Li et al., "BLIP: Bootstrapping Language-Image Pre-training," ICML 2022.
