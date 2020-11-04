# Face-Attribute-Prediction-v2  
Fae attribute prediction with multi-Task Learning.  

*UNDONE*

## To-Do  

- [x] Rewrite the training script of v1. 
- [x] Modify the approach of calculating the accuracy of classification.  
- [x] Re-train the v1-style network with ResNet-34 and new design of subtask.  
- [ ] Train the new network with **Uncertain Weight**.  
    > [1] Han, H., Jain, A. K., Wang, F., Shan, S., & Chen, X. (2018). Heterogeneous Face Attribute Estimation: A Deep Multi-Task Learning Approach. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(11), 2597–2609. https://doi.org/10.1109/TPAMI.2017.2738004  
  
- [ ] Train the new network with **Dynamic Weight Average**.
    > [2] Liu, S., Johns, E., & Davison, A. J. (2019). End-to-end multi-task learning with attention. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2019-June, 1871–1880. https://doi.org/10.1109/CVPR.2019.00197 


## Example  
1. Train  
    ```
    python mtl_main.py --pretrained --epochs 60 --batch-size 128 --lr 0.001 ...(training setting)
    ```  

2. Evaluate  
    ```
    python mtl_main.py --evaluate --resume checkpoint.pth
    ```  
