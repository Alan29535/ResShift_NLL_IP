We observe that texture regions exhibit higher uncertainty than smooth regions, making them more difficult for the model to learn. 
<img width="468" height="117" alt="image" src="https://github.com/user-attachments/assets/73b1024b-a43e-4e89-9c19-b632c2095426" />
To support this, we measure the KL divergence between each reverse step and its corresponding forward process on 16 ImageNet images.
<img width="391" height="194" alt="image" src="https://github.com/user-attachments/assets/63f1b5ba-ca2c-4c8b-8a66-47d3f9290d27" />
We further hypothesize that reconstruction error arises from the mismatch between the variance learned in the reverse process (via VLB) and the predefined variance in the forward process. To verify this, we track the variance evolution in both processes using ResShift (Fig. 3).
<img width="392" height="149" alt="image" src="https://github.com/user-attachments/assets/f8540d3a-96eb-4426-9350-d49fef0f8413" />



