# Comparative Benchmark Report: LUCAS

Generated: 2026-08-21 15:42:09

## EUNIS Class (CSV) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP32 | 47.7% | 46.0% | 45.3% | 54.2% | 59.3% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP16 | 47.7% | 46.0% | 45.3% | 54.2% | 59.3% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 47.1% | 45.4% | 44.8% | 53.8% | 58.7% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 47.1% | 45.4% | 44.8% | 53.8% | 58.6% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 46.7% | 45.5% | 45.1% | 54.2% | 58.9% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 46.7% | 45.5% | 45.1% | 54.2% | 58.9% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 44.8% | 43.8% | 43.8% | 51.9% | 56.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 44.8% | 43.8% | 43.8% | 51.9% | 56.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 44.7% | 44.1% | 43.8% | 52.1% | 56.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 44.7% | 44.1% | 43.8% | 52.1% | 56.7% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 44.6% | 43.7% | 43.3% | 51.6% | 56.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 44.6% | 43.9% | 43.6% | 52.5% | 57.1% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 44.6% | 43.9% | 43.7% | 52.5% | 57.1% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 44.5% | 43.4% | 43.1% | 51.4% | 56.4% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 44.5% | 43.7% | 43.3% | 51.6% | 56.3% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 44.5% | 43.4% | 43.1% | 51.4% | 56.4% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 44.3% | 43.4% | 43.4% | 51.5% | 56.2% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP32 | 44.3% | 43.2% | 43.1% | 51.8% | 56.7% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP16 | 44.3% | 43.2% | 43.1% | 51.8% | 56.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 44.2% | 44.1% | 43.8% | 52.3% | 56.9% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 44.2% | 44.1% | 43.8% | 52.3% | 56.9% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 44.2% | 43.4% | 43.4% | 51.5% | 56.2% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 43.9% | 43.8% | 43.7% | 51.5% | 55.9% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 43.9% | 43.8% | 43.7% | 51.5% | 55.9% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 43.7% | 42.7% | 42.7% | 51.4% | 56.4% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 43.7% | 42.7% | 42.7% | 51.4% | 56.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 43.4% | 43.6% | 43.3% | 51.6% | 55.9% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 43.4% | 43.6% | 43.3% | 51.6% | 55.9% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 43.1% | 42.1% | 42.2% | 50.8% | 55.4% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 43.0% | 42.1% | 42.2% | 50.8% | 55.4% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 43.0% | 43.6% | 43.2% | 51.7% | 56.0% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 43.0% | 43.6% | 43.2% | 51.7% | 56.0% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 40.2% | 39.4% | 39.2% | 48.7% | 53.4% |
| resnet50.a1_in1k | CLS + Avg Patch | FP32 | 40.2% | 39.4% | 39.1% | 48.7% | 53.4% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 40.2% | 39.4% | 39.1% | 48.7% | 53.4% |

## EUNIS Raster Level 1 (Macro) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 13.6% | 9.7% | 8.9% | 19.4% | 22.3% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 13.6% | 9.7% | 8.9% | 19.4% | 22.3% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 12.7% | 9.3% | 9.0% | 19.2% | 21.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 12.7% | 9.3% | 9.0% | 19.2% | 21.8% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 12.2% | 9.3% | 8.8% | 19.1% | 20.8% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP32 | 12.2% | 8.8% | 8.4% | 19.0% | 20.4% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 12.2% | 9.3% | 8.8% | 19.1% | 20.8% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP16 | 12.2% | 8.8% | 8.4% | 18.9% | 20.4% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 11.2% | 8.2% | 6.8% | 15.6% | 17.7% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 11.2% | 8.2% | 6.8% | 15.6% | 17.7% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 9.9% | 8.0% | 7.8% | 15.7% | 17.0% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 9.9% | 8.0% | 7.8% | 15.7% | 17.0% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 9.9% | 7.6% | 7.0% | 15.2% | 17.1% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 9.9% | 7.6% | 7.0% | 15.2% | 17.1% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 9.3% | 8.1% | 8.0% | 17.7% | 18.7% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 9.3% | 8.1% | 8.0% | 17.7% | 18.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 9.3% | 8.6% | 7.8% | 16.7% | 18.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 9.3% | 8.9% | 8.0% | 17.4% | 18.6% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 9.3% | 8.6% | 7.8% | 16.7% | 18.3% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 9.3% | 8.9% | 8.0% | 17.4% | 18.6% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP32 | 8.1% | 7.7% | 7.3% | 14.7% | 16.2% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP16 | 8.1% | 7.7% | 7.3% | 14.7% | 16.2% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 7.9% | 7.3% | 7.2% | 14.5% | 15.9% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 7.9% | 7.3% | 7.2% | 14.5% | 15.9% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 7.6% | 6.0% | 6.3% | 12.5% | 13.8% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 7.6% | 6.0% | 6.3% | 12.5% | 13.7% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 7.5% | 7.4% | 6.8% | 14.8% | 15.8% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 7.5% | 7.4% | 6.8% | 14.8% | 15.8% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 7.3% | 7.7% | 7.2% | 14.5% | 15.9% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 7.3% | 7.7% | 7.2% | 14.5% | 15.9% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 7.3% | 6.7% | 5.6% | 13.6% | 14.9% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 7.3% | 6.7% | 5.6% | 13.6% | 14.9% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 6.6% | 6.7% | 6.9% | 13.9% | 15.3% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 6.6% | 6.7% | 6.9% | 13.9% | 15.3% |

## EUNIS Raster Level 2 (Meso) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 11.1% | 7.9% | 7.2% | 16.2% | 18.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 11.1% | 7.6% | 7.4% | 16.1% | 18.6% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 11.1% | 7.9% | 7.2% | 16.2% | 18.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 11.1% | 7.6% | 7.4% | 16.1% | 18.6% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP32 | 10.8% | 7.6% | 7.0% | 16.2% | 17.5% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP16 | 10.8% | 7.6% | 7.0% | 16.2% | 17.5% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 10.5% | 7.5% | 6.0% | 14.7% | 16.2% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 10.5% | 7.9% | 7.5% | 16.0% | 17.6% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 10.5% | 7.5% | 6.0% | 14.7% | 16.2% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 10.5% | 7.9% | 7.5% | 16.0% | 17.6% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 8.9% | 6.7% | 6.4% | 13.3% | 14.6% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 8.9% | 6.7% | 6.4% | 13.3% | 14.6% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 8.4% | 6.7% | 6.8% | 15.7% | 16.6% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 8.4% | 6.7% | 6.8% | 15.7% | 16.6% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 8.4% | 5.7% | 5.7% | 12.1% | 13.6% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 8.4% | 5.7% | 5.7% | 12.1% | 13.6% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 7.5% | 7.3% | 6.4% | 13.6% | 14.9% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 7.5% | 7.4% | 6.6% | 14.2% | 15.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 7.5% | 7.3% | 6.4% | 13.6% | 14.9% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 7.5% | 7.4% | 6.6% | 14.2% | 15.2% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 7.3% | 5.6% | 5.8% | 11.7% | 12.8% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 7.3% | 5.6% | 5.8% | 11.7% | 12.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 7.3% | 5.9% | 5.9% | 12.4% | 13.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 7.3% | 5.9% | 5.9% | 12.4% | 13.8% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP32 | 7.2% | 6.2% | 6.0% | 12.7% | 14.1% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP16 | 7.2% | 6.2% | 6.0% | 12.7% | 14.1% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 6.0% | 5.2% | 4.5% | 11.5% | 12.4% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 6.0% | 5.2% | 4.5% | 11.5% | 12.4% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 5.7% | 6.1% | 5.6% | 12.3% | 13.0% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 5.7% | 6.1% | 5.6% | 12.3% | 13.0% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 5.0% | 6.0% | 5.7% | 11.2% | 11.9% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 5.0% | 6.0% | 5.7% | 11.2% | 11.9% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 4.3% | 5.5% | 5.4% | 10.5% | 11.2% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 4.3% | 5.5% | 5.4% | 10.5% | 11.2% |

## EUNIS Raster Level 3 (Exact) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP32 | 10.2% | 7.1% | 6.4% | 15.1% | 16.2% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP16 | 10.2% | 7.1% | 6.4% | 15.0% | 16.2% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 9.8% | 7.2% | 6.5% | 14.7% | 16.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 9.8% | 6.8% | 6.6% | 14.5% | 16.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 9.8% | 7.2% | 6.5% | 14.7% | 16.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 9.8% | 6.8% | 6.6% | 14.5% | 16.5% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 9.5% | 6.6% | 5.4% | 13.9% | 15.1% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 9.5% | 7.1% | 6.8% | 14.4% | 16.0% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 9.5% | 6.6% | 5.4% | 13.9% | 15.1% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 9.5% | 7.1% | 6.8% | 14.4% | 16.0% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 7.9% | 5.9% | 5.6% | 12.0% | 13.1% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 7.9% | 5.9% | 5.6% | 12.0% | 13.1% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 7.8% | 6.3% | 6.3% | 14.4% | 15.3% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 7.8% | 6.3% | 6.3% | 14.4% | 15.3% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 7.5% | 4.8% | 4.8% | 10.7% | 12.2% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 7.5% | 4.8% | 4.8% | 10.7% | 12.2% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 7.3% | 5.3% | 5.4% | 11.5% | 12.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 7.3% | 5.3% | 5.4% | 11.5% | 12.8% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP32 | 6.9% | 5.6% | 5.5% | 11.7% | 13.1% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP16 | 6.9% | 5.6% | 5.5% | 11.7% | 13.1% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 6.2% | 4.7% | 5.1% | 10.3% | 11.2% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 6.2% | 4.7% | 5.1% | 10.3% | 11.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 6.2% | 5.9% | 5.3% | 12.5% | 13.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 6.2% | 5.9% | 5.3% | 12.5% | 13.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 5.9% | 5.8% | 5.2% | 11.7% | 12.6% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 5.9% | 5.8% | 5.2% | 11.7% | 12.6% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 5.3% | 4.6% | 4.2% | 10.6% | 11.3% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 5.3% | 4.6% | 4.2% | 10.6% | 11.3% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 5.1% | 5.6% | 5.2% | 11.2% | 11.7% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 5.1% | 5.6% | 5.2% | 11.2% | 11.7% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 4.6% | 5.3% | 5.0% | 10.3% | 11.0% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 4.6% | 5.3% | 5.0% | 10.3% | 11.0% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 4.0% | 5.0% | 4.8% | 9.8% | 10.4% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 4.0% | 5.0% | 4.8% | 9.8% | 10.4% |

## Environmental Zones (Raster) Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 41.2% | 37.8% | 36.2% | 49.7% | 56.4% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 41.2% | 37.8% | 36.2% | 49.7% | 56.4% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 39.4% | 36.5% | 34.9% | 47.9% | 54.4% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 39.4% | 36.5% | 34.9% | 47.9% | 54.4% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 39.0% | 36.2% | 34.8% | 48.0% | 54.1% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 39.0% | 36.2% | 34.8% | 48.0% | 54.1% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP16 | 38.8% | 35.3% | 33.7% | 47.2% | 53.6% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP32 | 38.7% | 35.3% | 33.7% | 47.2% | 53.6% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP32 | 38.7% | 35.5% | 34.1% | 47.5% | 54.2% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP16 | 38.7% | 35.5% | 34.1% | 47.6% | 54.2% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 38.5% | 35.2% | 34.1% | 47.5% | 54.0% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 38.5% | 35.2% | 34.1% | 47.5% | 53.9% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 37.9% | 35.3% | 33.3% | 46.8% | 52.9% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 37.9% | 35.3% | 33.4% | 46.8% | 52.9% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 37.7% | 34.6% | 33.4% | 46.7% | 53.2% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 37.6% | 34.6% | 33.4% | 46.7% | 53.1% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 37.3% | 35.3% | 33.5% | 46.8% | 52.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 37.3% | 35.3% | 33.5% | 46.8% | 52.5% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 37.2% | 35.3% | 33.7% | 46.6% | 52.6% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 37.1% | 35.3% | 33.7% | 46.6% | 52.6% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 37.0% | 33.1% | 32.2% | 45.2% | 51.8% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 37.0% | 33.1% | 32.2% | 45.2% | 51.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 36.0% | 33.5% | 32.2% | 45.1% | 50.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 36.0% | 33.6% | 32.2% | 45.1% | 50.8% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 35.1% | 33.3% | 32.0% | 45.2% | 50.8% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 35.1% | 33.3% | 32.0% | 45.2% | 50.8% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 34.9% | 32.2% | 31.4% | 44.4% | 50.4% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 34.9% | 32.2% | 31.4% | 44.4% | 50.4% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 34.8% | 32.9% | 31.5% | 44.7% | 50.4% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 34.7% | 32.9% | 31.5% | 44.7% | 50.3% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 32.5% | 30.7% | 30.1% | 42.6% | 48.5% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 32.5% | 30.7% | 30.1% | 42.6% | 48.5% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 26.8% | 25.9% | 25.1% | 38.0% | 43.2% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 26.8% | 25.9% | 25.1% | 38.0% | 43.2% |

## Land Cover Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 44.9% | 43.3% | 42.5% | 51.3% | 55.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 44.9% | 43.3% | 42.5% | 51.3% | 55.7% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 44.1% | 42.6% | 41.9% | 50.4% | 54.6% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 44.1% | 42.6% | 41.9% | 50.4% | 54.6% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 44.0% | 42.4% | 41.3% | 50.3% | 54.6% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 44.0% | 42.4% | 41.4% | 50.3% | 54.6% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 43.9% | 42.6% | 41.8% | 51.0% | 55.1% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 43.9% | 42.6% | 41.8% | 51.0% | 55.1% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 43.6% | 41.9% | 41.1% | 50.0% | 54.4% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 43.6% | 41.9% | 41.1% | 50.0% | 54.4% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 43.2% | 42.2% | 41.4% | 49.9% | 54.0% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 43.2% | 42.1% | 41.4% | 49.9% | 54.0% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 43.1% | 42.0% | 41.4% | 50.5% | 54.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 43.1% | 42.0% | 41.4% | 50.5% | 54.4% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 42.8% | 42.2% | 41.6% | 49.7% | 53.5% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 42.8% | 42.2% | 41.6% | 49.7% | 53.5% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP32 | 42.7% | 41.9% | 40.9% | 49.9% | 54.0% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP16 | 42.7% | 41.9% | 40.9% | 49.9% | 54.0% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 42.6% | 41.7% | 41.2% | 49.7% | 53.7% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 42.6% | 41.7% | 41.2% | 49.7% | 53.7% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 42.5% | 41.5% | 40.4% | 49.3% | 53.6% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 42.5% | 41.5% | 40.4% | 49.3% | 53.6% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 41.8% | 40.6% | 39.8% | 48.8% | 53.2% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 41.8% | 40.6% | 39.8% | 48.8% | 53.2% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 39.2% | 38.2% | 37.6% | 46.5% | 50.9% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 39.2% | 38.1% | 37.6% | 46.5% | 50.9% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP32 | 38.5% | 37.0% | 36.4% | 45.7% | 50.1% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP16 | 38.5% | 37.0% | 36.4% | 45.7% | 50.1% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 37.8% | 36.3% | 35.4% | 44.9% | 49.4% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 37.8% | 36.3% | 35.4% | 44.9% | 49.4% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 37.5% | 37.7% | 37.2% | 45.8% | 49.6% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 37.5% | 37.7% | 37.2% | 45.8% | 49.6% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 33.2% | 31.6% | 30.8% | 40.7% | 45.3% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 33.1% | 31.6% | 30.8% | 40.7% | 45.3% |

## Land Use Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 69.9% | 68.4% | 67.8% | 73.5% | 76.8% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 69.8% | 68.4% | 67.8% | 73.5% | 76.8% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 69.1% | 68.0% | 67.6% | 73.1% | 76.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 69.1% | 68.0% | 67.6% | 73.1% | 76.2% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 68.9% | 68.6% | 68.0% | 73.2% | 76.3% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 68.9% | 68.6% | 67.9% | 73.2% | 76.3% |
| google/tipsv2-b14 | TIPSv2 CLS | FP32 | 68.7% | 68.1% | 67.9% | 73.0% | 76.2% |
| google/tipsv2-b14 | TIPSv2 CLS | FP16 | 68.7% | 68.1% | 67.9% | 73.0% | 76.2% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP32 | 68.6% | 68.1% | 67.5% | 72.8% | 75.9% |
| google/tipsv2-b14 | TIPSv2 CLS + Avg Patch | FP16 | 68.6% | 68.1% | 67.5% | 72.8% | 75.9% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 68.6% | 67.8% | 67.4% | 73.1% | 76.2% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 68.6% | 67.8% | 67.4% | 73.1% | 76.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 68.5% | 67.8% | 67.5% | 72.7% | 75.8% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 68.5% | 67.8% | 67.5% | 72.7% | 75.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 68.2% | 67.8% | 67.7% | 72.9% | 75.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 68.2% | 67.8% | 67.7% | 72.9% | 75.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 68.1% | 68.0% | 67.7% | 72.9% | 75.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 68.1% | 68.0% | 67.7% | 72.9% | 75.8% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP32 | 68.0% | 67.3% | 67.1% | 72.4% | 75.6% |
| google/tipsv2-b14 | TIPSv2 Average Patch | FP16 | 68.0% | 67.3% | 67.1% | 72.4% | 75.6% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 68.0% | 67.9% | 67.5% | 72.7% | 75.7% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 68.0% | 67.9% | 67.5% | 72.7% | 75.7% |
| vit_base_patch16_clip_224.openai | Average Patch | FP32 | 67.9% | 66.7% | 66.3% | 72.3% | 75.9% |
| vit_base_patch16_clip_224.openai | Average Patch | FP16 | 67.9% | 66.7% | 66.3% | 72.3% | 75.9% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP32 | 67.7% | 66.8% | 66.4% | 72.2% | 75.8% |
| vit_base_patch16_clip_224.openai | CLS + Avg Patch | FP16 | 67.7% | 66.8% | 66.4% | 72.2% | 75.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 67.6% | 68.0% | 67.7% | 72.8% | 75.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 67.6% | 68.0% | 67.7% | 72.8% | 75.5% |
| vit_base_patch16_clip_224.openai | CLS | FP32 | 67.2% | 66.4% | 66.1% | 71.8% | 75.4% |
| vit_base_patch16_clip_224.openai | CLS | FP16 | 67.2% | 66.4% | 66.1% | 71.8% | 75.4% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 65.9% | 65.6% | 65.7% | 70.7% | 73.8% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 65.9% | 65.6% | 65.7% | 70.7% | 73.9% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 64.5% | 63.8% | 63.7% | 69.5% | 73.3% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 64.5% | 63.8% | 63.7% | 69.5% | 73.3% |
