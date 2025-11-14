# Multiscale Adaptive Decoder and Diversity Selection Network for Road Extraction in Remote Sensing Image (TGRS 2025) #

## 1. Core Information
- **Title**: Multiscale Adaptive Decoder and Diversity Selection Network for Road Extraction in Remote Sensing Image  
- **Publication**: IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 63, 2025 
- **DOI**: 10.1109/TGRS.2025.3576200 
- **Authors**: Zhen-Tao Hua, Si-Bao Chen (Corresponding Author), Wei Lu, Jin Tang, Bin Luo 

## 2. Research Background & Problem
Road extraction from high-resolution remote sensing images (HRSIs) is critical for urban planning and disaster response, but existing methods face two key issues:  
1. Inadequate capture of contextual information, leading to fragmented or incomplete road segmentation;  
2. High false positives (FPs) and false negatives (FNs) when handling roads of varying sizes, especially under occlusions (e.g., trees, buildings) or complex backgrounds .

## 3. Proposed Method: MADSNet
A novel network integrating three core modules to address the above problems, based on an encoder-decoder architecture with ResNet34 as the pretrained backbone :
- **MFFE Decoder**: Combines the Relevance Inquiry Attention (RIA) module (window-based self-attention for long-range dependencies) and Scope Flexible Fusion (SFF) module (multiscale dilated convolutions to expand receptive field), enhancing context capture with low computational cost .  
- **OCGA Module**: Improves graph attention with KNN algorithm, aggregating neighboring nodes with similar features to strengthen focus on road regions and make up for MFFE’s local perception limitation .  
- **MFS Module**: Activates stage-specific road features and suppresses noise/interference across four decoder levels, fusing multiscale outputs to refine segmentation of roads of different sizes .

## 4. Experimental Validation
### 4.1 Datasets
- **Google Earth**: 224 images (1.2 m/pixel, 600×600 pixels), with complex backgrounds and occlusions ;  
- **Massachusetts**: 1171 aerial images (1 m/pixel, 1500×1500 pixels), covering cities/suburbs/villages ;  
- **CHN6-CUG**: 4511 images (0.5 m/pixel, 512×512 pixels) from 6 Chinese cities, with diverse data distribution .

### 4.2 Key Results
MADSNet outperforms state-of-the-art methods (e.g., U-Net, DeepLabV3+, CMTFNet) on all datasets, with representative metrics:  
| Dataset       | IoU    | F1-Score | OA      |
|---------------|--------|----------|---------|
| Google Earth  | 88.51% | 93.87%   | 98.47%  |
| Massachusetts | 66.14% | 78.72%   | 98.00%  |
| CHN6-CUG      | 59.62% | 71.82%   | 95.68%  | 

### 4.3 Advantages
- Excels in handling occluded roads (e.g., tree-covered paths) and similar background interference (e.g., red brick areas misclassified as roads by other methods) ;  
- Maintains road continuity and detail accuracy for dense road networks .

## 5. License & Acknowledgement
- **Funding**: Supported by NSFC Key Project (U20B2068), National Natural Science Foundation of China (62106006, 61976004) ;  
- **Code Repository**: https://github.com/Talent02/MADSNet .

## 6. Citation ##
If you find this code useful, please cite our paper.

```
@ARTICLE{11021615,
  author={Hua, Zhen-Tao and Chen, Si-Bao and Lu, Wei and Tang, Jin and Luo, Bin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multiscale Adaptive Decoder and Diversity Selection Network for Road Extraction in Remote Sensing Image}, 
  year={2025},
  volume={63},
  number={},
  pages={1-13},
  keywords={Feature extraction;Roads;Remote sensing;Transformers;Decoding;Convolution;Attention mechanisms;Deep learning;Data mining;Computer vision;Attention mechanism;feature selection;remote sensing images;road extraction},
  doi={10.1109/TGRS.2025.3576200}}
```
