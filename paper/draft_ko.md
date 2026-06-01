# Florence-2 기반 이미지 설명의 시각적 근거 검증 및 Hallucination 탐지

**Visual Evidence Verification for Image Caption Hallucination Detection using Florence-2**

> 본 파일은 IPIU2026 워크샵 양식(`Team/IPIU2026_sample/IPIU2026_sample.doc`)에 옮겨 담을 한국어 논문 원고 초안입니다. 분량 2~6쪽 (현재 추정 4쪽). 양식 규칙 — 제목 바탕체 14 bold, 본문 한글 바탕체 10 / 영문 Times New Roman 10, 장 번호 돋움체 12 bold, 표 캡션 상단·그림 캡션 하단 가운데 정렬, 발표자 이름 위에 `o` 표시.

---

## 요 약

본 연구는 통합 비전-언어 모델 Florence-2[1]의 captioning 결과를 같은 모델의 object detection 결과로 다시 검증하여, 이미지 설명 속 시각적 근거가 없는 객체 언급을 hallucination candidate로 탐지하는 self-verification 파이프라인을 제안한다. 파이프라인은 (1) `<CAPTION>` prompt로 caption을 생성하고, (2) COCO 80 어휘와 동의어 매핑으로 객체 mention을 추출한 뒤, (3) `<OD>` prompt 결과와 canonical 라벨 단위로 매칭하여 supported / unsupported로 분류하고, (4) COCO `instances_val2017` GT와 비교해 정량 평가한다. COCO val2017 50장에 대한 파일럿 실험에서, baseline caption의 hallucination rate 8.22%가 검증 후 2.90%로 약 65% 감소하였고, unsupported 객체 탐지 F1은 0.80, grounding accuracy@0.5는 97.01%를 기록하였다. 본 결과는 단일 통합 모델만으로도 caption-grounding consistency에 기반한 실용적 hallucination 검출이 가능함을 시사한다.

## 1. 서론

최근 Florence-2[1], BLIP-2, Qwen-VL 등 통합 비전-언어 모델(Vision-Language Model, VLM)이 자연스러운 이미지 설명을 생성한다. 그러나 이러한 모델은 실제 이미지에 존재하지 않는 객체를 caption에 포함하는 **object hallucination** 문제[3]를 보인다. 사용자가 caption만 보고 정보를 받아들이는 응용(접근성, 검색 인덱스, 검증된 자연어 인터페이스 등)에서 hallucination은 신뢰성을 직접적으로 저해한다.

대표적 평가 방법인 CHAIR[3]은 caption 속 객체 mention과 데이터셋 GT를 비교한다. 이는 평가에는 유용하지만, **사용 시점에 GT가 없는 환경에서는** 적용할 수 없다. 한편 grounding/detection 단계에서 분리된 모델을 사용하는 방식은 추가 모델 비용을 요구한다.

본 연구는 Florence-2가 captioning과 grounding을 모두 단일 prompt-conditioned 모델로 수행할 수 있다는 점에 주목하여, **별도 모델 없이 자신이 생성한 caption을 자신의 visual grounding 결과로 재검증하는** self-verification 파이프라인을 제안한다. 기여는 다음과 같다.

- Florence-2 단일 모델 위에서 동작하는 caption-grounding consistency 검증 파이프라인을 설계하고 구현한다.
- COCO 80 카테고리 어휘에 동의어 매핑을 결합한 mention 추출기를 제시한다.
- COCO val2017 50장 파일럿에서 hallucination rate를 8.22%→2.90%로 줄이고, unsupported 탐지 F1 0.80, grounding accuracy@0.5 97%를 달성한다.
- 동일 모델 self-verification의 한계(특히 caption과 OD의 *joint hallucination*)를 정성 사례로 분석한다.

본 연구의 목적은 hallucination의 완전 제거가 아니라 **candidate를 탐지하고 caption의 신뢰성·해석 가능성을 높이는** 것이다.

## 2. 관련 연구

**Florence-2[1]**는 captioning, object detection, phrase grounding, segmentation, OCR 등 다양한 비전·비전-언어 과제를 *task prompt*만 바꿔 단일 sequence-to-sequence 모델로 수행한다. FLD-5B라는 대규모 multi-task 어노테이션 데이터셋으로 학습되어 작은 모델 크기에도 강한 성능을 보인다. 본 연구는 `<CAPTION>`, `<DETAILED_CAPTION>`, `<OD>`, `<CAPTION_TO_PHRASE_GROUNDING>` prompt를 활용한다.

**Object hallucination 평가**. Rohrbach 등[3]은 CHAIR(Caption Hallucination Assessment with Image Relevance)를 도입해 caption mention이 GT 객체 집합에 포함되는지 평가하였다. POPE[4]는 polling 기반 평가로 hallucination을 더 큰 LVLM 환경에서 비교한다. 본 연구의 baseline 평가 지표는 CHAIR-like rate를 따른다.

**COCO 데이터셋[2]**. 80개 객체 카테고리와 이미지당 캡션·인스턴스 어노테이션을 제공하여 hallucination 평가에 사실상 표준으로 사용된다. 본 연구는 `val2017` 분할을 사용한다.

## 3. 방법

### 3.1 전체 파이프라인

본 시스템은 4단계 단방향 파이프라인이다(그림 1).

1. **Caption 생성**: Florence-2 `<CAPTION>` 또는 `<DETAILED_CAPTION>` prompt로 입력 이미지에 대한 caption을 생성한다.
2. **Object mention 추출**: COCO 80 카테고리 어휘 + 동의어 매핑 테이블을 사용해 caption에서 객체 mention을 추출한다.
3. **시각적 검증**: 같은 이미지에 `<OD>` prompt를 한 번 호출해 검출된 모든 (라벨, bbox) 쌍을 얻고, 각 mention의 canonical 라벨과 매칭한다.
4. **분류 및 시각화**: mention을 supported / unsupported로 분류한다. unsupported AND GT에도 부재하면 **hallucination candidate**로 표시한다.

### 3.2 Mention 추출 (Extraction)

caption 텍스트를 소문자화하고 영문/숫자/공백/하이픈 외 문자를 제거한 뒤, 80 canonical 라벨과 그 동의어를 펼친 lookup 테이블에 대해 *whole-word* 정규식 매칭을 수행한다. 다중-단어 표현(예: `dining table`, `fire hydrant`)이 단일 단어보다 먼저 매칭되도록 surface 길이 내림차순으로 정렬하고, 이미 매칭된 구간과 겹치는 후속 매칭은 폐기한다. 주요 동의어 매핑은 `man / woman / boy / girl / people → person`, `bike → bicycle`, `sofa → couch`, `television → tv`, `phone → cell phone` 등이다.

### 3.3 검증 (Verification)

`<OD>` 결과로 얻은 라벨 각각을 동일한 lookup 테이블로 canonical 라벨로 정규화한다. mention의 canonical과 detection의 canonical이 일치하면 supported이며, 첫 매칭의 bbox가 supported mention의 시각적 근거로 부여된다.

대안 전략으로 mention 1개당 `<CAPTION_TO_PHRASE_GROUNDING>`을 호출하는 방식도 구현했으나, 호출 횟수가 mention 수에 비례해 비용이 크고, `<OD>` 라벨 매칭과 결과 차이가 본 파일럿 규모에서 미미하여 본 보고서의 주된 보고에는 `<OD>` 전략을 채택했다.

### 3.4 정량 평가 지표

| 지표 | 정의 |
| --- | --- |
| Hallucination Rate (CHAIR-like) | (canonical ∉ image GT) / total mentions |
| Verified Hallucination Rate | (canonical ∉ GT) / supported mentions |
| Unsupported P / R / F1 | "unsupported로 깃발 들기" 작업에 대한 TP/FP/FN/TN 기반 |
| Grounding Acc@0.5 | supported bbox와 동일 카테고리 GT bbox의 max IoU ≥ 0.5인 비율 |

여기서 unsupported 탐지의 진리 라벨은 *canonical ∉ GT* 이다. 즉 TP는 시스템이 unsupported로 깃발 든 mention이 GT에도 정말로 없는 경우, FN은 시스템이 supported로 통과시켰지만 실제로 GT에 없는 경우(즉 시스템이 hallucination을 놓친 경우)이다.

## 4. 실험

### 4.1 데이터셋
COCO val2017 (5,000장)에서 시드(42)로 무작위 추출한 50장을 사용한다. 정답은 `instances_val2017.json` 의 80개 객체 카테고리이다.

### 4.2 구현 세부
PyTorch 2.5.1 + CUDA 12.1, NVIDIA RTX 3070. 모델은 `microsoft/Florence-2-base-ft` (base-ft, 약 230M parameters) 를 fp16으로 로드하여 inference만 수행한다. caption / OD prompt 모두 num_beams=3, do_sample=False. 첫 추론 시 모델 가중치 약 470MB가 Hugging Face에서 자동 다운로드된다. 본 연구는 추가 학습 없이 zero-shot inference만 사용한다.

### 4.3 Baseline
Florence-2 `<CAPTION>` 출력을 그대로 사용하고, mention 추출 결과를 별도 검증 없이 누적한다. 모든 mention이 "그대로 인정"되며 hallucination rate는 분모가 *total mentions* 이다.

### 4.4 제안 방법 (Ours, Verified Caption)
mention 추출 결과를 `<OD>`로 검증한 뒤 supported로 분류된 mention만 caption에 남긴다. 최종 verified caption은 unsupported mention을 제거한 형태이다.

## 5. 결과

### 5.1 정량 결과

표 1은 COCO val2017 50장(73 mentions)에 대한 결과이다.

**표 1. 50장 파일럿 결과** (Florence-2-base-ft)

| Method | Mentions | Hallucinated ↓ | Hall. Rate ↓ | Unsupp. F1 ↑ | Grd Acc@0.5 ↑ |
| --- | --: | --: | --: | --: | --: |
| `<CAPTION>` baseline | 73 | 6 | 8.22% | – | – |
| Ours (verified) | 69 | 2 | **2.90%** | **0.8000** | **0.9701** |

검증을 통해 hallucination rate가 약 65% 감소했다. unsupported 탐지의 confusion matrix는 TP=4, FP=0, FN=2, TN=67이며 precision은 1.0, recall은 0.667이다. supported로 통과된 mention 중 GT bbox와 IoU≥0.5인 비율은 97.01%로, 라벨 매칭이 단순 어휘 일치가 아닌 공간 정합을 동반함을 보였다.

### 5.2 정성 결과

**성공 사례 1 (그림 2)** — `image_id 6012`, caption *"A bunch of bananas and an apple in a bowl."*: GT는 `banana`만. 시스템은 caption의 `bowl`을 unsupported로 정확히 분류했다 (TP). 그러나 동일 이미지의 `apple` 은 `<OD>`도 같은 hallucination을 공유하여 supported로 통과되었다 (FN). 이 한 이미지에서 success와 failure가 함께 나타나는 점이 self-verification의 양면을 직접 보여준다.

**성공 사례 2 (그림 3)** — `image_id 2153`, caption *"A baseball player holding a bat on top of a field."*: GT에 `sports ball` 은 없고 `baseball bat`, `person`만 존재. 시스템은 caption의 *baseball*을 정확히 unsupported로 처리했다. 비슷한 야구 장면(`6471`, `10764`)에서도 동일하게 동작했다.

**실패 사례 (그림 4)** — `image_id 5503`, caption *"A person standing in front of a toilet with the seat up."*: GT에는 `toilet` 만 있고 사람은 없으나 시스템은 `person`을 supported로 통과시켰다. 이는 caption과 `<OD>`가 동일 방향으로 hallucinate한 joint failure이다.

## 6. 논의

### 6.1 해석
검증으로 hallucination rate가 8.22%→2.90%로 떨어진 것은 작은 표본이지만 일관된 신호이다. precision이 1.0이라는 점은 시스템이 *과도 기각하지 않음*을 의미하며, recall 0.667은 *추가 개선 여지가 있음*을 의미한다.

### 6.2 한계
1. **Self-verification의 본질**: caption과 grounding을 같은 Florence-2 모델로 수행하므로 동일 방향 오류는 검출되지 않는다 (그림 4의 toilet/person joint failure가 직접 증거).
2. **어휘 한정**: COCO 80 카테고리 밖의 객체(예: 본 실험에서 0-mention 처리된 *tripod*, *plate*, *rice*)는 추출되지 않는다.
3. **표기 모호성**: caption은 *baseball*, *table*, *phone* 같은 표현을 쓰지만 `<OD>`는 `baseball bat`, `dining table`, `cell phone` 등 더 좁은 표기를 사용한다. 동의어 매핑이 완벽하지 못한 부분이 unsupported 4건 중 3건에서 노출되었다.
4. **단일 보정 형태**: verified caption은 unsupported mention을 *제거*만 한다. 더 부드러운 대안 (예: 신뢰도 표시, 부분 보정) 은 향후 과제.

### 6.3 향후 개선 방향
- caption 모델과 검증 모델을 분리 (예: BLIP-2 caption + Florence-2 OD) 하여 joint failure 완화.
- SAM2 등 segmentation을 결합해 bbox보다 정밀한 visual evidence 제공.
- attribute (색·재질) 및 relation (앞/뒤/들고 있는) hallucination까지 확장.
- POPE, FOIL 등 더 큰 benchmark로 비교.
- 한국어 caption verification으로 확장.

## 7. 결론

본 연구는 Florence-2의 통합 multi-task 능력을 활용해 caption-grounding consistency를 검증하는 self-verification 파이프라인을 제안하였다. COCO val2017 50장 파일럿에서 hallucination rate는 8.22%에서 2.90%로 약 65% 감소했으며, unsupported 탐지 F1 0.80, grounding accuracy@0.5 97%를 보였다. 동일 모델 self-verification의 한계가 정성 분석에서 명확히 드러났고, 이 한계는 검증 모델 분리·정밀한 visual evidence·관계 hallucination 확장 등 명확한 후속 연구 방향으로 이어진다.

## 감사의 글

본 연구는 컴퓨터비전 과목 프로젝트의 일환으로 수행되었다.

## 참고문헌

[1] B. Xiao et al., "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks," CVPR 2024.
[2] T. Y. Lin et al., "Microsoft COCO: Common Objects in Context," ECCV 2014.
[3] A. Rohrbach et al., "Object Hallucination in Image Captioning," EMNLP 2018.
[4] Y. Li et al., "Evaluating Object Hallucination in Large Vision-Language Models," EMNLP 2023.
