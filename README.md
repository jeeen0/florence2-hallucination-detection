# Florence-2 기반 이미지 설명의 시각적 근거 검증 및 Hallucination 탐지

**Visual Evidence Verification for Image Caption Hallucination Detection using Florence-2**

Florence-2 통합 비전-언어 모델의 captioning 결과를 같은 모델의 detection / phrase grounding 결과로 다시 검증하여, caption 내 시각적 근거가 없는 객체 언급을 *hallucination candidate* 로 표시·정정하는 self-verification 파이프라인.

---

## Overview

최신 Vision-Language Model은 자연스러운 이미지 설명을 생성하지만, 실제 이미지에 존재하지 않는 객체를 포함하는 **object hallucination** 문제가 발생할 수 있다. 본 프로젝트는 Florence-2의 multi-task capability(captioning + grounding + detection)를 활용하여, 모델이 생성한 caption을 다시 grounding 결과로 검증하는 **caption-grounding consistency pipeline**을 제안한다.

> Vision-language model의 caption은 자연스럽게 보일 수 있지만, 항상 이미지에 근거한 것은 아니다. Florence-2처럼 captioning과 grounding을 모두 수행할 수 있는 모델을 활용하면, caption 속 객체 언급이 실제 이미지에 시각적 근거를 갖는지 확인할 수 있다.

## Pipeline

```
Input Image
    → Florence-2 <CAPTION> / <DETAILED_CAPTION>
    → Object Mention Extraction (COCO80 vocabulary + synonym mapping)
    → Florence-2 <OD>  or  <CAPTION_TO_PHRASE_GROUNDING>
    → Supported / Unsupported classification
    → Visualization + Verified Caption
```

### Object classification (3-class)

| Class | Meaning |
| --- | --- |
| **Supported** | caption mention is grounded/detected in the image |
| **Unsupported** | caption mention is NOT grounded/detected |
| **Hallucination Candidate** | Unsupported AND absent from COCO GT |

## Target Paper

Xiao, Bin, et al. *Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.* CVPR 2024.

## Method

### 1. Caption Generation
Florence-2의 `<CAPTION>` / `<DETAILED_CAPTION>` prompt로 입력 이미지에 대한 caption 생성.

### 2. Object Mention Extraction
생성된 caption에서 검증할 객체 명사를 COCO 80 category 기준으로 추출. 동의어/표기 변이는 매핑 테이블로 정규화.

Example synonym mapping:
```
man / woman / people / boy / girl  →  person
bike                                →  bicycle
sofa                                →  couch
tv / television                     →  tv
motorbike                           →  motorcycle
```

### 3. Visual Verification
Florence-2의 `<OD>` (object detection) 또는 `<CAPTION_TO_PHRASE_GROUNDING>` 으로 caption 객체가 실제 이미지에 존재하는지 검증.

### 4. Decision Rule
검증 결과를 supported / unsupported / hallucination candidate 3-class로 분류하고, 시각화 + verified caption 생성.

## Experiments

### Dataset
- **COCO val2017**
  - Development: 20–50 images
  - Final evaluation: 100–300 images
  - Qualitative cases: 3 success + 2 failure examples

### Models
| Use | Model |
| --- | --- |
| Development / small-scale | `microsoft/Florence-2-base-ft` |
| Final / main experiments | `microsoft/Florence-2-large-ft` |

### Metrics
- **Object Hallucination Rate** = hallucinated mentions / total mentions
- **CHAIR-like Rate** = mentions not in GT / total mentions
- **Unsupported Object Detection**: Precision / Recall / F1
- **Grounding Acc@0.5** (IoU ≥ 0.5; falls back to image-level category presence if time-constrained)

### Result table skeleton

| Method | Mentions | Hallucinated ↓ | Hall. Rate ↓ | Unsupp. F1 ↑ | Grd Acc@0.5 ↑ |
| --- | --: | --: | --: | --: | --: |
| Florence-2 Caption (Baseline) | – | – | – | – | – |
| Ours Verified Caption         | – | – | – | – | – |

## Repository structure

```
florence2-hallucination-detection/
├── README.md
├── .gitignore
├── code/
│   ├── env/                 # conda env / requirements
│   ├── florence2/           # model loader wrapper
│   ├── modules/
│   │   ├── caption.py       # <CAPTION> / <DETAILED_CAPTION>
│   │   ├── extract.py       # object mention + synonym mapping
│   │   ├── verify.py        # <OD> / phrase grounding verification
│   │   ├── classify.py      # supported / unsupported / hallucination
│   │   └── visualize.py     # bbox + verified caption visualization
│   └── pipeline.py          # CLI entry point
├── data/
│   ├── coco_val/            # input images (gitignored)
│   └── coco_annotations/    # instances_val2017.json etc. (gitignored)
├── outputs/
│   ├── captions/            # caption + extracted-object CSVs
│   ├── grounding/           # verification result CSVs
│   ├── visualizations/      # bbox images
│   └── metrics/             # *_metrics.json, *_result_table.csv
└── paper/                   # IPIU2026 manuscript + figures
```

## Environment

- **Python** 3.10
- **PyTorch** + CUDA
- Transformers, Accelerate, Timm, Einops, Pillow, OpenCV, Matplotlib, Pandas, NumPy, Tqdm, Pycocotools, PyYAML

### GPU plan
| GPU | Use |
| --- | --- |
| RTX 3070 | base-ft development & small-scale experiments |
| RTX 4090 | large-ft final experiments |
| RTX PRO 6000 | (if available) large-ft final & bulk processing |

## Quickstart

```bash
# 1. Create env (conda)
conda env create -f code/env/environment.yml
conda activate florence2-hallu

# Or with pip:
# pip install -r code/env/requirements.txt

# 2. Step 1 smoke test — run <CAPTION> / <DETAILED_CAPTION> / <OD> on one image
python code/run_sample.py --image path/to/image.jpg

# Outputs:
#   outputs/captions/sample_caption.csv
#   outputs/captions/sample_od.csv
#   outputs/captions/sample_raw.json
#   outputs/visualizations/sample_od.jpg
```

First run downloads Florence-2 weights from Hugging Face (~470 MB for `base-ft`). Set `--model microsoft/Florence-2-large-ft` for the larger checkpoint.

## Roadmap

| Step | Focus | Key artifacts |
| --: | --- | --- |
| 1 | conda env, Florence-2-base-ft 실행, `<CAPTION>`/`<OD>` 출력 형식 확정 | `outputs/captions/sample_caption.csv`, `outputs/visualizations/sample_od.jpg` |
| 2 | COCO val 일부 caption 생성 + mention 추출 (synonym 매핑 포함) | `outputs/captions/base_caption_50.csv`, `..._extracted_objects.csv` |
| 3 | object별 grounding/detection 검증 + bbox 시각화 | `outputs/grounding/base_grounding_results.csv`, `outputs/visualizations/base/` |
| 4 | COCO GT로 Hall. Rate / P·R·F1 / (가능 시) IoU@0.5 계산 | `outputs/metrics/base_metrics.json`, `..._result_table.csv` |
| 5 | base-ft 50–100장 정성 결과 (성공 3 / 실패 2), Method·Experiment 초안 | qualitative figures, draft text |
| 6 | RTX 4090에서 Florence-2-large-ft 100–300장 최종 실험 | `outputs/metrics/large_metrics.json`, `outputs/visualizations/large/` |
| 7 | IPIU2026 양식 최종보고서 + 제출 패키지 | `paper/` 산출물 |

## Claim

> Florence-2가 captioning과 grounding을 동시에 수행할 수 있다는 점을 활용하면, 별도 모델 없이 self-verification 파이프라인을 구성할 수 있다. 본 연구는 caption-grounding consistency 검증을 통해 unsupported object mention을 탐지하고, 이미지 설명의 신뢰성과 해석 가능성을 높이는 실용적 파이프라인을 제시한다.

목표는 hallucination의 완전 제거가 아니라 **candidate 탐지 / consistency 검증**이다.

## Limitations

1. Caption 생성과 grounding 검증에 동일한 Florence-2 계열 모델을 사용하므로 완전히 독립적인 검증은 아니다.
2. Grounding 실패가 실제 hallucination으로 오탐될 수 있다.
3. COCO 80 category 기반 객체 추출은 caption 속 모든 객체를 포괄하지 못한다.
4. Caption correction은 단순히 unsupported object를 제거하는 수준으로 제한된다.
5. 복잡한 관계·속성 hallucination은 본 범위를 벗어난다.

## Future Work

1. Caption 생성 모델과 grounding 검증 모델을 분리하여 독립적인 verification 수행.
2. SAM2 등 segmentation 모델을 결합하여 box보다 정밀한 visual evidence 제공.
3. Attribute / relation hallucination까지 확장.
4. CHAIR, POPE 등 기존 hallucination metric과 더 큰 benchmark에서 비교.
5. 한국어 및 multilingual caption verification으로 확장.

## References

1. Xiao, Bin, et al. *Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.* CVPR 2024.
2. Lin, Tsung-Yi, et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014.
3. Rohrbach, Anna, et al. *Object Hallucination in Image Captioning.* EMNLP 2018.
4. Li, Yifan, et al. *Evaluating Object Hallucination in Large Vision-Language Models (POPE).* EMNLP 2023.
