Observations and Analysis

We observe that texture regions exhibit higher uncertainty than smooth regions, making them more difficult for the model to learn.
The corresponding implementation can be found in reprod_results2 copy.ipynb.

<p align="center"> <img src="https://github.com/user-attachments/assets/73b1024b-a43e-4e89-9c19-b632c2095426" width="468"/> </p>

To further support this observation, we measure the KL divergence between each reverse step and its corresponding forward process using 16 ImageNet images.

<p align="center"> <img src="https://github.com/user-attachments/assets/63f1b5ba-ca2c-4c8b-8a66-47d3f9290d27" width="391"/> </p>

We further hypothesize that the reconstruction error arises from the mismatch between the variance learned in the reverse process (optimized via VLB) and the predefined variance in the forward process.
To verify this, we track the variance evolution in both processes using ResShift. The implementation is available in variance_compare_three_models.ipynb.

<p align="center"> <img src="https://github.com/user-attachments/assets/f8540d3a-96eb-4426-9350-d49fef0f8413" width="392"/> </p>
