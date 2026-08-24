# Comparative Benchmark Report: PLACES
Generated: 2026-08-20 11:57:41

## Exact Place Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| google/tipsv2-b14 | CLS | FP32 | 37.7% | 34.5% | 32.9% | 44.6% | 49.9% |
| google/tipsv2-b14 | CLS | FP16 | 37.7% | 34.5% | 32.9% | 44.6% | 49.9% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 36.6% | 33.7% | 32.3% | 43.1% | 47.6% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 36.6% | 33.7% | 32.3% | 43.1% | 47.6% |
| google/tipsv2-b14 | CLS + Avg Patch | FP32 | 36.2% | 33.2% | 31.3% | 43.4% | 48.4% |
| google/tipsv2-b14 | CLS + Avg Patch | FP16 | 36.2% | 33.2% | 31.3% | 43.4% | 48.4% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 36.2% | 34.1% | 32.3% | 43.3% | 47.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 36.2% | 34.5% | 32.5% | 43.6% | 48.0% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 36.2% | 34.1% | 32.3% | 43.3% | 47.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 36.2% | 34.5% | 32.5% | 43.6% | 48.0% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 36.2% | 34.1% | 32.3% | 43.3% | 47.8% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 36.2% | 34.5% | 32.5% | 43.6% | 48.0% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 36.2% | 34.1% | 32.3% | 43.3% | 47.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 36.2% | 34.5% | 32.5% | 43.6% | 48.0% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 35.4% | 32.9% | 31.9% | 42.2% | 46.5% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 35.4% | 32.9% | 31.9% | 42.2% | 46.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 34.4% | 31.7% | 30.0% | 41.7% | 46.2% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 34.4% | 31.7% | 30.0% | 41.7% | 46.2% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 34.4% | 31.7% | 30.0% | 41.7% | 46.2% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 34.4% | 31.7% | 30.0% | 41.7% | 46.2% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 34.2% | 31.0% | 29.5% | 41.2% | 46.1% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 34.2% | 31.0% | 29.5% | 41.2% | 46.1% |
| google/tipsv2-b14 | Average Patch | FP32 | 34.1% | 30.5% | 28.2% | 40.4% | 45.6% |
| google/tipsv2-b14 | Average Patch | FP16 | 34.1% | 30.5% | 28.2% | 40.4% | 45.6% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 34.0% | 31.9% | 30.0% | 41.1% | 44.8% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 34.0% | 31.9% | 30.0% | 41.1% | 44.8% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 25.3% | 22.5% | 21.0% | 33.4% | 37.3% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 25.3% | 22.5% | 21.0% | 33.4% | 37.3% |

## Macro Category Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| google/tipsv2-b14 | CLS | FP32 | 85.9% | 84.7% | 84.4% | 87.8% | 90.7% |
| google/tipsv2-b14 | CLS | FP16 | 85.9% | 84.7% | 84.4% | 87.8% | 90.7% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 85.5% | 83.6% | 83.2% | 87.0% | 90.2% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 85.5% | 83.6% | 83.2% | 87.0% | 90.2% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 85.4% | 84.3% | 83.8% | 87.5% | 90.5% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 85.4% | 84.3% | 83.8% | 87.5% | 90.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 85.4% | 84.2% | 83.6% | 87.4% | 90.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 85.4% | 84.2% | 83.6% | 87.4% | 90.5% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 85.4% | 84.2% | 83.6% | 87.4% | 90.5% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 85.4% | 84.2% | 83.6% | 87.4% | 90.5% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 85.0% | 84.3% | 83.7% | 87.4% | 90.3% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 85.0% | 84.3% | 83.7% | 87.4% | 90.3% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 84.9% | 83.1% | 82.3% | 86.5% | 89.8% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 84.9% | 83.1% | 82.3% | 86.5% | 89.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 84.9% | 84.3% | 83.7% | 87.4% | 90.2% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 84.9% | 84.3% | 83.7% | 87.4% | 90.2% |
| google/tipsv2-b14 | CLS + Avg Patch | FP32 | 84.8% | 84.5% | 83.9% | 87.6% | 90.2% |
| google/tipsv2-b14 | CLS + Avg Patch | FP16 | 84.8% | 84.5% | 83.9% | 87.6% | 90.2% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 84.8% | 83.8% | 83.3% | 87.1% | 90.1% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 84.8% | 83.8% | 83.3% | 87.1% | 90.1% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 84.8% | 83.8% | 83.3% | 87.1% | 90.1% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 84.8% | 83.8% | 83.3% | 87.1% | 90.1% |
| google/tipsv2-b14 | Average Patch | FP32 | 83.9% | 83.2% | 82.6% | 86.5% | 89.6% |
| google/tipsv2-b14 | Average Patch | FP16 | 83.9% | 83.2% | 82.6% | 86.5% | 89.6% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 83.7% | 81.8% | 80.9% | 85.7% | 89.1% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 83.6% | 81.8% | 80.9% | 85.7% | 89.1% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 82.6% | 81.3% | 80.5% | 85.2% | 88.6% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 82.6% | 81.3% | 80.5% | 85.2% | 88.6% |

## Sub-Category Comparison

| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| google/tipsv2-b14 | CLS | FP16 | 63.1% | 61.2% | 60.3% | 68.4% | 73.7% |
| google/tipsv2-b14 | CLS | FP32 | 63.0% | 61.2% | 60.3% | 68.4% | 73.7% |
| google/tipsv2-b14 | CLS + Avg Patch | FP16 | 62.2% | 59.9% | 58.4% | 67.5% | 72.8% |
| google/tipsv2-b14 | CLS + Avg Patch | FP32 | 62.1% | 59.9% | 58.4% | 67.5% | 72.8% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP32 | 61.7% | 59.6% | 58.7% | 66.7% | 71.8% |
| vit_base_patch14_dinov2.lvd142m | CLS + Avg Patch | FP16 | 61.7% | 59.6% | 58.7% | 66.7% | 71.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP32 | 61.5% | 59.9% | 58.8% | 67.1% | 72.2% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS + Avg Patch | FP16 | 61.5% | 59.9% | 58.8% | 67.2% | 72.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP32 | 61.5% | 59.9% | 58.8% | 67.1% | 72.2% |
| vit_base_patch16_dinov3.lvd1689m | CLS + Avg Patch | FP16 | 61.5% | 59.9% | 58.8% | 67.2% | 72.2% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP32 | 61.1% | 58.4% | 57.3% | 66.4% | 72.0% |
| vit_base_patch14_dinov2.lvd142m | Average Patch | FP16 | 61.1% | 58.4% | 57.3% | 66.4% | 72.0% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP32 | 60.9% | 59.6% | 58.4% | 66.9% | 71.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | CLS | FP16 | 60.9% | 59.6% | 58.4% | 66.9% | 71.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP32 | 60.9% | 59.6% | 58.4% | 66.9% | 71.7% |
| vit_base_patch16_dinov3.lvd1689m | CLS | FP16 | 60.9% | 59.6% | 58.4% | 66.9% | 71.7% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP32 | 60.8% | 58.5% | 57.0% | 66.3% | 71.8% |
| vit_base_patch16_dinov3_qkvb.lvd1689m | Average Patch | FP16 | 60.8% | 58.5% | 57.0% | 66.3% | 71.8% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP32 | 60.8% | 58.5% | 57.0% | 66.3% | 71.8% |
| vit_base_patch16_dinov3.lvd1689m | Average Patch | FP16 | 60.8% | 58.5% | 57.0% | 66.3% | 71.8% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP32 | 60.5% | 58.9% | 57.8% | 66.0% | 70.9% |
| vit_base_patch14_dinov2.lvd142m | CLS | FP16 | 60.5% | 58.9% | 57.8% | 66.0% | 70.9% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP16 | 59.2% | 57.2% | 55.7% | 64.9% | 69.9% |
| convnext_base.dinov3_lvd1689m | Average (No CLS) | FP32 | 59.1% | 57.2% | 55.7% | 64.9% | 69.9% |
| google/tipsv2-b14 | Average Patch | FP32 | 59.0% | 57.2% | 55.4% | 64.7% | 70.4% |
| google/tipsv2-b14 | Average Patch | FP16 | 59.0% | 57.2% | 55.4% | 64.7% | 70.4% |
| resnet50.a1_in1k | Average (No CLS) | FP32 | 53.2% | 50.1% | 48.6% | 59.7% | 66.0% |
| resnet50.a1_in1k | Average (No CLS) | FP16 | 53.2% | 50.1% | 48.6% | 59.7% | 66.0% |
