# Comparative Benchmark Report: EUNIS
Generated: 2026-08-24 02:23:54

## EUNIS Level 1 (Macro) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP32 | 17.0% | 16.0% | 15.5% | 24.8% | 28.1% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP16 | 17.0% | 16.0% | 15.5% | 24.9% | 28.1% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 16.8% | 16.5% | 16.3% | 24.7% | 27.6% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP32 | 16.8% | 16.5% | 16.3% | 24.7% | 27.6% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 16.8% | 16.5% | 16.3% | 24.7% | 27.6% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP16 | 16.8% | 16.5% | 16.3% | 24.7% | 27.6% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP32 | 16.8% | 16.2% | 16.0% | 24.2% | 27.0% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP16 | 16.8% | 16.2% | 16.0% | 24.2% | 27.0% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 16.6% | 16.3% | 16.1% | 24.1% | 26.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 16.6% | 16.3% | 16.1% | 24.1% | 26.8% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP32 | 16.5% | 15.2% | 14.8% | 23.1% | 26.1% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP16 | 16.5% | 15.2% | 14.8% | 23.1% | 26.1% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 16.3% | 15.8% | 15.6% | 24.4% | 27.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 16.3% | 15.8% | 15.6% | 24.4% | 27.5% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 16.2% | 15.2% | 14.8% | 23.0% | 25.8% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 16.2% | 15.2% | 14.8% | 23.0% | 25.8% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP32 | 16.0% | 16.2% | 15.9% | 24.2% | 26.8% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP16 | 16.0% | 16.2% | 15.9% | 24.2% | 26.8% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP32 | 15.6% | 15.0% | 14.8% | 22.6% | 25.4% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP16 | 15.6% | 15.0% | 14.8% | 22.6% | 25.3% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 15.4% | 15.1% | 14.9% | 22.5% | 25.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 15.4% | 15.1% | 14.9% | 22.5% | 25.2% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP32 | 15.3% | 14.6% | 14.3% | 22.7% | 25.6% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP16 | 15.3% | 14.6% | 14.3% | 22.7% | 25.6% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP32 | 15.2% | 14.7% | 14.6% | 22.6% | 25.4% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP16 | 15.2% | 14.7% | 14.6% | 22.6% | 25.4% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP32 | 15.0% | 14.1% | 13.9% | 22.4% | 25.3% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP16 | 15.0% | 14.1% | 13.9% | 22.4% | 25.3% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP16 | 14.8% | 14.8% | 14.5% | 22.9% | 25.4% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP32 | 14.8% | 14.5% | 14.2% | 22.7% | 25.6% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP16 | 14.8% | 14.5% | 14.2% | 22.7% | 25.6% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 14.7% | 14.7% | 14.5% | 23.2% | 25.8% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP32 | 14.7% | 14.8% | 14.4% | 22.9% | 25.4% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 14.7% | 14.7% | 14.5% | 23.2% | 25.8% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 14.6% | 14.5% | 14.2% | 23.0% | 25.6% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP32 | 14.6% | 14.4% | 13.9% | 23.1% | 25.5% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 14.6% | 14.5% | 14.2% | 23.0% | 25.6% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP16 | 14.6% | 14.4% | 13.9% | 23.1% | 25.5% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 14.6% | 14.7% | 14.5% | 22.6% | 25.2% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 14.6% | 14.7% | 14.5% | 22.6% | 25.2% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 14.5% | 14.3% | 14.3% | 22.5% | 25.2% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 14.4% | 14.3% | 14.3% | 22.5% | 25.2% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 14.4% | 14.6% | 14.2% | 22.8% | 25.5% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 14.4% | 14.6% | 14.2% | 22.8% | 25.5% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 14.2% | 14.3% | 14.3% | 22.3% | 24.8% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 14.2% | 14.3% | 14.3% | 22.3% | 24.8% |
| resnet50.a1_in1k | Seg-Masked | FP32 | 14.0% | 14.0% | 14.4% | 23.4% | 26.2% |
| resnet50.a1_in1k | Seg-Masked | FP16 | 14.0% | 14.0% | 14.4% | 23.3% | 26.2% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 13.8% | 14.3% | 14.5% | 23.4% | 26.2% |
| resnet50.a1_in1k | CLS + Average Patch | FP32 | 13.8% | 14.3% | 14.5% | 23.4% | 26.2% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 13.8% | 14.3% | 14.5% | 23.4% | 26.2% |
| resnet50.a1_in1k | CLS + Average Patch | FP16 | 13.8% | 14.3% | 14.5% | 23.4% | 26.2% |

## EUNIS Level 2 (Meso) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP32 | 7.4% | 7.0% | 6.9% | 12.1% | 13.2% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP16 | 7.4% | 7.0% | 6.9% | 12.1% | 13.2% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 7.1% | 7.0% | 7.1% | 11.8% | 13.0% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP32 | 7.1% | 7.0% | 7.1% | 11.8% | 13.0% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 7.1% | 7.0% | 7.1% | 11.8% | 13.0% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP16 | 7.1% | 7.0% | 7.1% | 11.8% | 13.0% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP32 | 6.9% | 6.5% | 6.3% | 11.3% | 12.6% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP16 | 6.9% | 6.5% | 6.3% | 11.3% | 12.6% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 6.8% | 6.2% | 6.1% | 10.6% | 11.7% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 6.8% | 6.2% | 6.1% | 10.6% | 11.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP32 | 6.7% | 6.5% | 6.3% | 10.9% | 11.9% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP16 | 6.7% | 6.5% | 6.3% | 10.9% | 11.9% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP32 | 6.6% | 6.1% | 5.8% | 10.7% | 11.7% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP16 | 6.6% | 6.1% | 5.8% | 10.7% | 11.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP32 | 6.6% | 6.4% | 6.3% | 10.7% | 11.9% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP16 | 6.6% | 6.4% | 6.3% | 10.7% | 11.8% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP32 | 6.5% | 6.3% | 6.1% | 10.7% | 11.7% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP16 | 6.5% | 6.3% | 6.1% | 10.7% | 11.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 6.5% | 6.5% | 6.4% | 10.8% | 11.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 6.5% | 6.5% | 6.4% | 10.8% | 11.8% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 6.5% | 6.4% | 6.4% | 10.4% | 11.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 6.5% | 6.4% | 6.4% | 10.4% | 11.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 6.4% | 6.3% | 6.2% | 11.3% | 12.3% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 6.4% | 6.3% | 6.2% | 11.2% | 12.3% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP32 | 6.3% | 5.9% | 5.9% | 10.5% | 11.7% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP32 | 6.3% | 5.9% | 5.8% | 10.6% | 11.8% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP16 | 6.3% | 5.9% | 5.9% | 10.5% | 11.7% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP16 | 6.3% | 5.9% | 5.8% | 10.6% | 11.8% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP32 | 6.2% | 6.1% | 6.0% | 10.6% | 11.6% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP16 | 6.2% | 6.1% | 6.0% | 10.6% | 11.7% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 6.1% | 5.7% | 5.8% | 10.6% | 11.6% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 6.1% | 5.7% | 5.8% | 10.5% | 11.5% |
| resnet50.a1_in1k | Seg-Masked | FP32 | 6.1% | 5.7% | 5.9% | 10.7% | 11.8% |
| resnet50.a1_in1k | Seg-Masked | FP16 | 6.1% | 5.7% | 5.9% | 10.7% | 11.8% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP32 | 6.1% | 5.7% | 5.6% | 10.2% | 11.3% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP16 | 6.1% | 5.7% | 5.6% | 10.2% | 11.3% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 6.0% | 6.3% | 6.1% | 10.8% | 11.9% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 6.0% | 6.3% | 6.1% | 10.8% | 11.9% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP32 | 6.0% | 5.7% | 5.5% | 10.4% | 11.5% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP16 | 6.0% | 5.7% | 5.5% | 10.4% | 11.5% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 5.9% | 6.1% | 6.2% | 10.8% | 11.9% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 5.9% | 6.1% | 6.2% | 10.8% | 11.9% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 5.9% | 5.9% | 5.8% | 10.3% | 11.2% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 5.9% | 5.9% | 5.8% | 10.3% | 11.2% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 5.9% | 5.7% | 5.5% | 10.4% | 11.4% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 5.9% | 5.7% | 5.5% | 10.3% | 11.4% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 5.8% | 5.8% | 5.9% | 10.6% | 11.7% |
| resnet50.a1_in1k | CLS + Average Patch | FP32 | 5.8% | 5.8% | 5.9% | 10.6% | 11.7% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 5.8% | 5.8% | 5.9% | 10.6% | 11.7% |
| resnet50.a1_in1k | CLS + Average Patch | FP16 | 5.8% | 5.8% | 5.9% | 10.6% | 11.7% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 5.8% | 5.7% | 5.5% | 10.1% | 11.1% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 5.8% | 5.7% | 5.5% | 10.1% | 11.1% |

## EUNIS Level 3 (Exact) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 3.2% | 2.8% | 2.6% | 5.0% | 5.5% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP32 | 3.2% | 2.8% | 2.7% | 5.3% | 5.7% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 3.2% | 2.8% | 2.6% | 5.0% | 5.5% |
| vit_base_patch14_dinov2.lvd142m | CLS + Average Patch | FP16 | 3.2% | 2.8% | 2.7% | 5.3% | 5.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP32 | 3.0% | 2.7% | 2.6% | 5.2% | 5.6% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Seg-Masked | FP16 | 3.0% | 2.7% | 2.6% | 5.1% | 5.6% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 2.9% | 2.4% | 2.3% | 5.0% | 5.3% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP32 | 2.9% | 2.5% | 2.4% | 4.9% | 5.4% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 2.9% | 2.4% | 2.3% | 4.9% | 5.3% |
| google/tipsv2-b14 | TIPSv2 CLS + Average Patch | FP16 | 2.9% | 2.5% | 2.4% | 4.9% | 5.4% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP32 | 2.9% | 2.9% | 2.8% | 5.2% | 5.6% |
| convnext_base.dinov3_lvd1689m | Seg-Masked | FP16 | 2.9% | 2.9% | 2.8% | 5.2% | 5.6% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 2.9% | 2.7% | 2.6% | 5.3% | 5.7% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 2.9% | 2.7% | 2.5% | 5.2% | 5.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 2.8% | 2.7% | 2.6% | 5.2% | 5.6% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 2.8% | 2.7% | 2.6% | 5.0% | 5.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 2.8% | 2.7% | 2.6% | 5.2% | 5.6% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 2.8% | 2.7% | 2.6% | 5.0% | 5.4% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP32 | 2.8% | 2.6% | 2.5% | 5.2% | 5.6% |
| vit_base_patch16_clip_224.openai | Seg-Masked | FP16 | 2.8% | 2.6% | 2.5% | 5.2% | 5.6% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 2.7% | 2.7% | 2.6% | 5.2% | 5.6% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 2.7% | 2.7% | 2.6% | 5.2% | 5.6% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 2.7% | 2.8% | 2.8% | 5.1% | 5.5% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP32 | 2.7% | 2.8% | 2.8% | 5.1% | 5.5% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 2.7% | 2.8% | 2.8% | 5.1% | 5.5% |
| convnext_base.dinov3_lvd1689m | CLS + Average Patch | FP16 | 2.7% | 2.8% | 2.8% | 5.1% | 5.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP32 | 2.7% | 2.8% | 2.6% | 5.0% | 5.3% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Average Patch | FP16 | 2.7% | 2.8% | 2.6% | 5.0% | 5.3% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 2.7% | 2.8% | 2.7% | 4.7% | 5.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 2.7% | 2.8% | 2.7% | 4.7% | 5.2% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP32 | 2.6% | 2.4% | 2.4% | 5.0% | 5.4% |
| google/tipsv2-b14 | TIPSv2 Seg-Masked | FP16 | 2.6% | 2.4% | 2.4% | 5.0% | 5.4% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 2.6% | 2.6% | 2.6% | 4.8% | 5.1% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP32 | 2.6% | 2.7% | 2.6% | 4.8% | 5.1% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 2.6% | 2.6% | 2.6% | 4.8% | 5.1% |
| vit_base_patch14_dinov2.lvd142m | Seg-Masked | FP16 | 2.6% | 2.7% | 2.6% | 4.8% | 5.1% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP32 | 2.6% | 2.5% | 2.5% | 5.1% | 5.5% |
| vit_base_patch16_clip_224.openai | CLS + Average Patch | FP16 | 2.6% | 2.5% | 2.5% | 5.1% | 5.5% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP32 | 2.5% | 2.8% | 2.7% | 4.7% | 5.1% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP32 | 2.5% | 2.5% | 2.4% | 4.6% | 4.9% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Average Patch | FP16 | 2.5% | 2.8% | 2.7% | 4.7% | 5.1% |
| vit_base_patch16_dinov3.lvd1689m | Seg-Masked | FP16 | 2.5% | 2.5% | 2.4% | 4.6% | 4.9% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 2.4% | 2.6% | 2.5% | 5.0% | 5.3% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 2.4% | 2.6% | 2.5% | 5.0% | 5.3% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 2.2% | 2.6% | 2.6% | 4.6% | 4.9% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 2.2% | 2.6% | 2.6% | 4.6% | 4.9% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 2.0% | 1.9% | 1.9% | 3.8% | 4.2% |
| resnet50.a1_in1k | CLS + Average Patch | FP32 | 2.0% | 1.9% | 1.9% | 3.8% | 4.2% |
| resnet50.a1_in1k | Seg-Masked | FP32 | 2.0% | 2.0% | 1.9% | 3.9% | 4.2% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 2.0% | 1.9% | 1.9% | 3.8% | 4.2% |
| resnet50.a1_in1k | CLS + Average Patch | FP16 | 2.0% | 1.9% | 1.9% | 3.8% | 4.2% |
| resnet50.a1_in1k | Seg-Masked | FP16 | 2.0% | 2.0% | 1.9% | 3.9% | 4.2% |
