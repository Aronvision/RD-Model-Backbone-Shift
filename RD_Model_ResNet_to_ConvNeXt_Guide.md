# RD 모델 ResNet → ConvNeXt 변환 가이드

이 문서는 기존 ResNet 백본을 사용하던 RD 모델을 ConvNeXt 기반으로 변경하기 위한 단계별 안내서입니다.  
(대화에서 정리한 내용을 토대로 작성)

---

## 1. 전체 구조 이해
- **ResNet (백본)**: 입력 이미지를 점차 다운샘플하며 feature 추출
- **BN Layer (브릿지)**: stage별 feature 채널/해상도 정렬 + concat
- **de_resnet (디코더)**: 브릿지 출력 → 업샘플 → 최종 출력

ConvNeXt로 변경 시 **ResNet 백본 부분만 ConvNeXt로 교체**하고, BN Layer와 디코더는 채널·해상도만 맞춰 그대로 재사용.

---

## 2. BN Layer의 역할
- ResNet stage별 feature의 **채널/해상도 정렬**
- BatchNorm + Conv 등을 이용해 안정적 feature 융합
- 최종적으로 디코더가 사용할 하나의 큰 feature 맵 생성

이 역할은 ConvNeXt 백본에도 동일하게 적용됩니다.

---

## 3. ConvNeXt 백본 준비
1. torchvision 또는 timm에서 ConvNeXt Tiny/Small/Base 불러오기  
2. `forward()`에서 stage별 feature(예: stride 4/8/16) 3개 추출  
3. 1×1 Conv + BN으로 채널을 ResNet과 동일하게 맞춤 (예: 64/128/256)  
4. `[feature1, feature2, feature3]` 리스트 반환

```python
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn

class ConvNeXtBackbone(nn.Module):
    def __init__(self, out_channels=(64,128,256), pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        feats = [96,192,384]
        self.proj1 = nn.Conv2d(feats[0], out_channels[0], 1, bias=False)
        self.proj2 = nn.Conv2d(feats[1], out_channels[1], 1, bias=False)
        self.proj3 = nn.Conv2d(feats[2], out_channels[2], 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feats = []
        h = x
        for i, block in enumerate(self.backbone.features):
            h = block(h)
            if i == 1: f1 = self.relu(self.bn1(self.proj1(h)))
            if i == 2: f2 = self.relu(self.bn2(self.proj2(h)))
            if i == 3: f3 = self.relu(self.bn3(self.proj3(h)))
        return [f1, f2, f3]
````

---

## 4. 파일 구조 제안

```
models/
  resnet_backbone.py
  convnext_backbone.py   ← 신규 작성
  bn_layer.py
  de_resnet.py
  de_convnext.py         ← ConvNeXt 전용 디코더(업샘플)
```

* **Teacher(ConvNeXt)**: Pretrained Weight 그대로 → `forward()`에서 feature만 추출
* **BN Layer**: 그대로 사용하되 채널/stride만 ConvNeXt에 맞게 조정
* **de\_convnext.py**: `de_resnet.py` 참고해 업샘플 ConvTranspose2d 버전 작성

---

## 5. 디코더 전환 (해상도 늘리기)

* `de_resnet`에서 쓰던 downsampling Conv를 `ConvTranspose2d`로 바꿔 업샘플링 구현
* 채널 수는 BN Layer 최종 출력 채널에 맞춤
* 필요 없는 모듈(FC/classifier)은 제거

---

## 6. 학습/평가 순서

1. **Baseline**: ResNet 기반 모델 학습+평가(3회 평균)
2. **ConvNeXt**: 백본 교체 후 스모크 테스트(출력 shape 확인)
3. **동일 설정으로 학습**: lr/wd/sched 동일하게 맞춰 비교
4. **ConvNeXt 최적 설정 학습**: AdamW, WD 0.05, cosine lr 등 튜닝
5. **테스트셋 평가**: 동일 프로토콜/seed로 측정, 통계 비교
6. **오류 분석 및 리포트 작성**

---

## 7. 팁 정리 (이미지 메모 해석)

* Teacher(ConvNeXt)는 **구조 변경 금지**, `forward()`에서 feature만 tap
* ResNet.py와 분리해 **convnext\_backbone.py**와 **de\_convnext.py**를 따로 만들기
* **CNBlock** 재사용해 채널/stride만 변경
* 디코더에서 **해상도 줄이는 것 → 늘리는 것**으로 바꾸기
* 불필요 부분 삭제해 단순화

---

## 8. 실험 비교 체크리스트

* 동일 데이터/시드/전처리/증강
* 동일 디코더·손실·학습시간
* 주요 지표(mIoU, mAP 등) 및 FLOPs/Params 기록
* 학습곡선, 오류케이스 시각화

---

## 9. 최종 요약

* **백본 변경 핵심**: ResNet→ConvNeXt시 feature shape·채널만 맞추면 BN Layer/디코더 재사용 가능
* **파일 구조 분리**: 유지보수/실험 용이
* **학습 비교**: Baseline→ConvNeXt(동일셋팅)→ConvNeXt(튜닝셋팅) 순으로 진행
* **평가 분석**: 지표+곡선+케이스별 분석으로 결론 도출

이 과정을 따르면 RD모델의 백본을 ConvNeXt로 안전하게 교체하고 공정한 성능 비교가 가능합니다.

```

래
