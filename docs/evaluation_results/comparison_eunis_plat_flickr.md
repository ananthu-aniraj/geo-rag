# Comparative Benchmark Report: EUNIS

Generated: 2026-09-06 04:44:53

## EUNIS Level 1 (Macro) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 18.8% | 17.0% | 16.5% | 26.9% | 30.3% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP32 | 18.8% | 17.0% | 16.5% | 26.9% | 30.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 18.8% | 17.0% | 16.5% | 26.9% | 30.3% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP16 | 18.8% | 17.0% | 16.5% | 26.9% | 30.2% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP32 | 18.0% | 17.2% | 16.9% | 26.9% | 30.3% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP16 | 18.0% | 17.2% | 16.9% | 26.9% | 30.3% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP32 | 17.9% | 17.1% | 16.6% | 26.7% | 29.9% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP16 | 17.9% | 17.1% | 16.6% | 26.7% | 29.9% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 17.9% | 17.2% | 16.8% | 26.9% | 30.2% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 17.9% | 17.2% | 16.8% | 26.9% | 30.2% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP32 | 17.8% | 16.6% | 16.5% | 26.3% | 29.6% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP16 | 17.8% | 16.6% | 16.5% | 26.3% | 29.6% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 17.7% | 17.1% | 16.6% | 26.8% | 29.8% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 17.7% | 17.1% | 16.6% | 26.8% | 29.8% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP32 | 17.3% | 16.4% | 15.8% | 26.2% | 29.4% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP16 | 17.3% | 16.4% | 15.8% | 26.2% | 29.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 17.3% | 16.7% | 16.5% | 26.3% | 29.3% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 17.3% | 16.7% | 16.6% | 26.3% | 29.3% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 17.3% | 17.3% | 16.8% | 26.9% | 30.0% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP32 | 17.3% | 17.0% | 16.6% | 26.5% | 29.7% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 17.3% | 17.3% | 16.8% | 26.9% | 30.0% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP16 | 17.3% | 17.0% | 16.6% | 26.5% | 29.7% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 17.1% | 16.0% | 16.0% | 25.8% | 29.0% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP32 | 17.1% | 16.0% | 16.0% | 25.8% | 29.0% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 17.1% | 16.0% | 16.0% | 25.8% | 29.0% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP16 | 17.1% | 16.0% | 16.0% | 25.8% | 29.0% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP32 | 16.9% | 15.9% | 15.8% | 25.3% | 28.8% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP16 | 16.9% | 15.9% | 15.8% | 25.3% | 28.8% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP32 | 16.9% | 16.3% | 16.3% | 25.9% | 29.1% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP16 | 16.9% | 16.3% | 16.3% | 25.9% | 29.1% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP32 | 16.7% | 16.2% | 16.1% | 26.0% | 29.0% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP16 | 16.7% | 16.2% | 16.1% | 26.0% | 29.0% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP32 | 16.6% | 16.2% | 16.1% | 26.0% | 29.1% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP16 | 16.6% | 16.2% | 16.1% | 26.0% | 29.1% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 16.5% | 16.3% | 16.2% | 25.6% | 28.7% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 16.5% | 16.3% | 16.2% | 25.6% | 28.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP16 | 16.5% | 16.1% | 15.7% | 25.6% | 28.8% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 16.4% | 15.7% | 15.6% | 25.4% | 28.7% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 16.4% | 15.7% | 15.6% | 25.4% | 28.7% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 16.4% | 16.1% | 16.0% | 25.8% | 28.9% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 16.4% | 16.1% | 16.0% | 25.8% | 28.9% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP32 | 16.4% | 16.1% | 15.7% | 25.6% | 28.8% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 16.4% | 16.0% | 16.1% | 25.7% | 28.6% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 16.4% | 16.0% | 16.1% | 25.7% | 28.6% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 16.1% | 15.8% | 15.7% | 25.4% | 28.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 16.1% | 15.8% | 15.7% | 25.4% | 28.5% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 14.6% | 14.8% | 14.7% | 24.8% | 27.6% |
| resnet50.a1_in1k | CLS + Average Patch | FP32 | 14.6% | 14.8% | 14.7% | 24.8% | 27.6% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 14.6% | 14.8% | 14.7% | 24.8% | 27.6% |
| resnet50.a1_in1k | CLS + Average Patch | FP16 | 14.6% | 14.8% | 14.7% | 24.8% | 27.6% |
| resnet50.a1_in1k | Seg-Masked | FP32 | 14.2% | 14.3% | 14.4% | 24.0% | 26.9% |
| resnet50.a1_in1k | Seg-Masked | FP16 | 14.2% | 14.3% | 14.4% | 24.0% | 26.9% |

## EUNIS Level 2 (Meso) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 9.9% | 8.5% | 8.3% | 14.9% | 16.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP32 | 9.9% | 8.6% | 8.3% | 15.0% | 16.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 9.9% | 8.5% | 8.3% | 14.9% | 16.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP16 | 9.9% | 8.6% | 8.3% | 15.0% | 16.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP32 | 9.0% | 8.3% | 8.3% | 14.5% | 16.1% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP16 | 9.0% | 8.3% | 8.3% | 14.5% | 16.1% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 8.9% | 8.3% | 8.1% | 14.7% | 16.2% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 8.9% | 8.3% | 8.1% | 14.7% | 16.3% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 8.8% | 7.8% | 7.8% | 14.3% | 15.9% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 8.8% | 7.8% | 7.8% | 14.3% | 15.9% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP32 | 8.8% | 8.4% | 8.2% | 14.7% | 16.4% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP16 | 8.8% | 8.4% | 8.2% | 14.7% | 16.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 8.7% | 8.3% | 8.2% | 14.3% | 15.9% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 8.7% | 8.3% | 8.2% | 14.3% | 15.9% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP32 | 8.5% | 7.9% | 7.7% | 14.1% | 15.7% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP16 | 8.5% | 7.9% | 7.7% | 14.1% | 15.7% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 8.5% | 8.3% | 8.2% | 14.8% | 16.2% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 8.5% | 8.3% | 8.2% | 14.8% | 16.2% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP32 | 8.4% | 8.4% | 8.2% | 14.7% | 16.2% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP16 | 8.4% | 8.4% | 8.2% | 14.7% | 16.2% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 8.1% | 7.5% | 7.4% | 13.3% | 14.7% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 8.1% | 7.5% | 7.4% | 13.3% | 14.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 8.1% | 7.8% | 7.7% | 13.9% | 15.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP32 | 8.1% | 7.9% | 7.6% | 13.8% | 15.3% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 8.1% | 7.8% | 7.7% | 14.0% | 15.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP16 | 8.1% | 7.9% | 7.6% | 13.8% | 15.3% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP32 | 8.0% | 7.7% | 7.3% | 13.3% | 14.6% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP16 | 8.0% | 7.7% | 7.3% | 13.3% | 14.6% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 7.8% | 7.3% | 7.3% | 13.2% | 14.6% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP32 | 7.8% | 7.3% | 7.3% | 13.2% | 14.6% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 7.8% | 7.3% | 7.3% | 13.2% | 14.6% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP16 | 7.8% | 7.3% | 7.3% | 13.2% | 14.6% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP32 | 7.7% | 7.3% | 7.1% | 13.1% | 14.3% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP16 | 7.7% | 7.3% | 7.1% | 13.1% | 14.3% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 7.5% | 7.3% | 7.2% | 12.9% | 14.2% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 7.5% | 7.3% | 7.2% | 12.9% | 14.2% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 7.5% | 7.2% | 7.0% | 13.0% | 14.2% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 7.5% | 7.2% | 7.0% | 13.0% | 14.2% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP32 | 7.4% | 7.3% | 7.0% | 13.1% | 14.3% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP16 | 7.4% | 7.3% | 7.0% | 13.1% | 14.3% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP32 | 7.3% | 7.1% | 7.1% | 12.6% | 13.9% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP16 | 7.3% | 7.1% | 7.1% | 12.6% | 13.9% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 7.1% | 6.8% | 6.9% | 12.5% | 13.8% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 7.1% | 6.8% | 6.9% | 12.5% | 13.8% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP32 | 7.1% | 7.3% | 7.4% | 13.1% | 14.2% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP16 | 7.1% | 7.3% | 7.4% | 13.1% | 14.2% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 6.5% | 6.4% | 6.3% | 12.1% | 13.3% |
| resnet50.a1_in1k | CLS + Average Patch | FP32 | 6.5% | 6.4% | 6.3% | 12.1% | 13.3% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 6.5% | 6.4% | 6.3% | 12.1% | 13.3% |
| resnet50.a1_in1k | CLS + Average Patch | FP16 | 6.5% | 6.4% | 6.3% | 12.1% | 13.3% |
| resnet50.a1_in1k | Seg-Masked | FP32 | 6.3% | 6.2% | 6.2% | 11.8% | 13.0% |
| resnet50.a1_in1k | Seg-Masked | FP16 | 6.3% | 6.2% | 6.2% | 11.8% | 13.0% |

## EUNIS Level 3 (Exact) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 4.7% | 4.0% | 4.0% | 7.6% | 8.4% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 4.7% | 4.0% | 4.0% | 7.6% | 8.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 4.5% | 4.2% | 4.1% | 7.8% | 8.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP32 | 4.5% | 4.1% | 4.2% | 7.8% | 8.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 4.5% | 4.2% | 4.1% | 7.8% | 8.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP16 | 4.5% | 4.1% | 4.2% | 7.8% | 8.5% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP32 | 4.5% | 4.0% | 4.0% | 7.7% | 8.3% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP16 | 4.5% | 4.0% | 4.0% | 7.7% | 8.3% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 4.4% | 3.8% | 3.8% | 7.4% | 8.1% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 4.4% | 3.8% | 3.8% | 7.5% | 8.2% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 4.4% | 3.7% | 3.6% | 7.2% | 7.9% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 4.4% | 3.7% | 3.6% | 7.2% | 8.0% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 4.2% | 4.0% | 3.8% | 7.6% | 8.3% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP32 | 4.2% | 4.1% | 3.9% | 7.6% | 8.4% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 4.2% | 4.0% | 3.8% | 7.6% | 8.3% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP16 | 4.2% | 4.1% | 3.9% | 7.6% | 8.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP32 | 4.1% | 3.8% | 3.7% | 7.2% | 8.0% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP16 | 4.1% | 3.8% | 3.7% | 7.2% | 8.0% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP32 | 4.1% | 3.8% | 3.7% | 7.5% | 8.1% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP16 | 4.1% | 3.8% | 3.7% | 7.5% | 8.1% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP32 | 3.9% | 3.5% | 3.4% | 6.6% | 7.3% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP16 | 3.9% | 3.5% | 3.4% | 6.6% | 7.3% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 3.9% | 3.4% | 3.4% | 6.9% | 7.5% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP32 | 3.9% | 3.4% | 3.4% | 6.9% | 7.5% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 3.9% | 3.4% | 3.4% | 6.9% | 7.5% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP16 | 3.9% | 3.4% | 3.4% | 6.9% | 7.5% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP32 | 3.9% | 3.5% | 3.5% | 6.9% | 7.6% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP16 | 3.9% | 3.5% | 3.5% | 6.9% | 7.6% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 3.9% | 3.8% | 3.7% | 7.4% | 8.0% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 3.9% | 3.8% | 3.7% | 7.4% | 8.0% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 3.8% | 3.3% | 3.2% | 6.5% | 7.2% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 3.8% | 3.3% | 3.2% | 6.5% | 7.2% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP32 | 3.8% | 3.4% | 3.2% | 6.3% | 6.9% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP16 | 3.8% | 3.4% | 3.2% | 6.3% | 6.9% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 3.7% | 3.5% | 3.4% | 6.6% | 7.1% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 3.7% | 3.5% | 3.4% | 6.6% | 7.1% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 3.7% | 3.2% | 3.2% | 6.2% | 6.8% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 3.7% | 3.2% | 3.2% | 6.2% | 6.8% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP32 | 3.6% | 3.4% | 3.3% | 6.6% | 7.1% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP16 | 3.6% | 3.4% | 3.3% | 6.6% | 7.1% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 3.5% | 3.1% | 3.0% | 6.1% | 6.6% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 3.5% | 3.1% | 3.0% | 6.1% | 6.6% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP32 | 3.5% | 3.4% | 3.4% | 6.7% | 7.2% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP16 | 3.5% | 3.4% | 3.4% | 6.7% | 7.2% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP32 | 3.3% | 3.1% | 3.0% | 6.1% | 6.6% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP16 | 3.3% | 3.1% | 3.0% | 6.1% | 6.6% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 3.3% | 2.8% | 2.7% | 5.7% | 6.2% |
| resnet50.a1_in1k | CLS + Average Patch | FP32 | 3.3% | 2.8% | 2.7% | 5.7% | 6.2% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 3.3% | 2.8% | 2.7% | 5.7% | 6.2% |
| resnet50.a1_in1k | CLS + Average Patch | FP16 | 3.3% | 2.8% | 2.7% | 5.7% | 6.2% |
| resnet50.a1_in1k | Seg-Masked | FP32 | 3.1% | 2.7% | 2.6% | 5.4% | 5.9% |
| resnet50.a1_in1k | Seg-Masked | FP16 | 3.1% | 2.7% | 2.6% | 5.4% | 5.9% |

## Interactive Retrieval Visualizers

| Model | Visualizer Dashboard |
| :--- | :--- |
| google/tipsv2-b14 | [eunis_visualizer_google_tipsv2-b14_s42_q5000_plat-flickr.html](eunis_visualizer_google_tipsv2-b14_s42_q5000_plat-flickr.html) |
| vit_base_patch14_dinov2.lvd142m | [eunis_visualizer_vit_base_patch14_dinov2.lvd142m_s42_q5000_plat-flickr.html](eunis_visualizer_vit_base_patch14_dinov2.lvd142m_s42_q5000_plat-flickr.html) |
| resnet50.a1_in1k | [eunis_visualizer_resnet50.a1_in1k_s42_q5000_plat-flickr.html](eunis_visualizer_resnet50.a1_in1k_s42_q5000_plat-flickr.html) |
| convnext_base.dinov3_lvd1689m | [eunis_visualizer_convnext_base.dinov3_lvd1689m_s42_q5000_plat-flickr.html](eunis_visualizer_convnext_base.dinov3_lvd1689m_s42_q5000_plat-flickr.html) |
| vit_base_patch16_dinov3_qkvb.lvd1689m | [eunis_visualizer_vit_base_patch16_dinov3_qkvb.lvd1689m_s42_q5000_plat-flickr.html](eunis_visualizer_vit_base_patch16_dinov3_qkvb.lvd1689m_s42_q5000_plat-flickr.html) |
| vit_base_patch16_dinov3.lvd1689m | [eunis_visualizer_vit_base_patch16_dinov3.lvd1689m_s42_q5000_plat-flickr.html](eunis_visualizer_vit_base_patch16_dinov3.lvd1689m_s42_q5000_plat-flickr.html) |
| vit_base_patch16_clip_224.openai | [eunis_visualizer_vit_base_patch16_clip_224.openai_s42_q5000_plat-flickr.html](eunis_visualizer_vit_base_patch16_clip_224.openai_s42_q5000_plat-flickr.html) |
